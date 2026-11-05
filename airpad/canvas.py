"""Stroke model + raster layers with undo / redo / clear / export.

Ink is stored in two layers of the display size:

* ``color`` - premultiplied ink colour (BGR, uint8)
* ``inv``   - inverse ink coverage (255 = no ink), replicated over 3 channels

Anti-aliased drawing with OpenCV blends ``new = old * (1 - c) + value * c``
where ``c`` is pixel coverage. Drawing the ink colour into ``color`` and 0
into ``inv`` is therefore exactly the premultiplied *over* operator, and
drawing (0, 255) is exactly *destination-out* (an eraser). Compositing over
any background is just ``out = color + bg * inv / 255``: two saturating uint8
OpenCV calls, and only inside the bounding box of the ink.

Live strokes are rendered incrementally. A spline segment depends on its four
neighbouring points, so once a few newer points exist its pixels can never
change: those segments are *frozen* straight into the committed layers and
only the last few segments (the tail) are redrawn per frame. Per-frame cost is
therefore constant, however long the stroke. Freeze boundaries are recorded on
the stroke, so re-rendering (undo/redo) reproduces the live pixels exactly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .spline import catmull_rom, smooth_slow_segments

_SHIFT = 4  # sub-pixel precision for OpenCV drawing (coordinates * 16)
_ONE = 1 << _SHIFT

Rect = tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)


def _union(a: Rect | None, b: Rect | None) -> Rect | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


@dataclass
class Stroke:
    color: tuple[int, int, int]
    size: int
    erase: bool = False
    points: list[tuple[float, float]] = field(default_factory=list)
    # Raw-segment indices where frozen chunks ended during live drawing.
    chunks: list[int] = field(default_factory=list)

    def add(self, x: float, y: float, min_dist: float = 1.0) -> bool:
        """Append a point unless it is within ``min_dist`` px of the last one."""
        if self.points:
            lx, ly = self.points[-1]
            if (x - lx) ** 2 + (y - ly) ** 2 < min_dist * min_dist:
                return False
        self.points.append((float(x), float(y)))
        return True


class _Painter:
    """Rasterises one stroke's geometry into (color, inv) layer pairs."""

    def __init__(self, stroke: Stroke, scale: float):
        self.stroke = stroke
        self.scale = scale
        self.size = max(1, int(stroke.size))
        if stroke.erase:
            self.ink, self.cov = (0, 0, 0), (255, 255, 255)
        else:
            self.ink, self.cov = tuple(int(c) for c in stroke.color), (0, 0, 0)
        self.spacing = float(np.clip(self.size * 0.5, 2.0, 6.0))
        self.margin = self.size // 2 + 3

    def samples(self, points, seg_from: int) -> tuple[np.ndarray, np.ndarray]:
        """Spline samples for raw segments ``seg_from`` .. end of ``pts``.

        Returns ``(samples, offsets)``: samples of global segment ``seg_from + k``
        are ``samples[offsets[k]:offsets[k + 1]]`` and the last sample is the
        stroke's end point. Each segment only depends on nearby points and is
        computed with element-wise maths, so results are bit-identical no
        matter which window of the stroke they were computed from.
        """
        # Smoothing needs 1 neighbour, the spline needs 1 more on each side.
        w0 = max(0, seg_from - 3)
        window = np.asarray(points[w0:], dtype=np.float64)
        window = smooth_slow_segments(window, dense=4.0 * self.scale, sparse=12.0 * self.scale)
        samples, offsets = catmull_rom(window, spacing=self.spacing, remove_duplicates=False,
                                       return_offsets=True)
        k = seg_from - w0
        start = offsets[k]
        return samples[start:], offsets[k:] - start

    def polyline(self, color_img, inv_img, pts: np.ndarray) -> Rect:
        ipts = np.round(pts * _ONE).astype(np.int32)
        for img, val in ((color_img, self.ink), (inv_img, self.cov)):
            cv2.polylines(img, [ipts.reshape(-1, 1, 2)], False, val, self.size, cv2.LINE_AA, _SHIFT)
        lo = np.floor(pts.min(axis=0)).astype(int) - self.margin
        hi = np.ceil(pts.max(axis=0)).astype(int) + self.margin + 1
        return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])

    def dot(self, color_img, inv_img, p) -> Rect:
        c = (int(round(p[0] * _ONE)), int(round(p[1] * _ONE)))
        r = max(1, int(round(self.size / 2 * _ONE)))
        for img, val in ((color_img, self.ink), (inv_img, self.cov)):
            cv2.circle(img, c, r, val, -1, cv2.LINE_AA, _SHIFT)
        m = self.margin
        return int(p[0]) - m, int(p[1]) - m, int(p[0]) + m + 2, int(p[1]) + m + 2

    def draw_range(self, color_img, inv_img, points, seg_from: int, seg_to: int | None) -> Rect | None:
        """Draw raw segments [seg_from, seg_to) (``None`` = to the end) as one polyline."""
        n = len(points)
        if n == 1:
            return self.dot(color_img, inv_img, points[0])
        if n == 2:
            return self.polyline(color_img, inv_img, np.asarray(points, dtype=np.float64))
        samples, offsets = self.samples(points, seg_from)
        end = len(samples) if seg_to is None else offsets[seg_to - seg_from] + 1
        return self.polyline(color_img, inv_img, samples[:end])

    def render_full(self, color_img, inv_img) -> Rect | None:
        """Re-render a finished stroke, replaying the live freeze boundaries."""
        pts = np.asarray(self.stroke.points, dtype=np.float64)
        if len(pts) == 0:
            return None
        if len(pts) < 3:
            return self.draw_range(color_img, inv_img, self.stroke.points, 0, None)
        samples, offsets = catmull_rom(
            smooth_slow_segments(pts, dense=4.0 * self.scale, sparse=12.0 * self.scale),
            spacing=self.spacing, remove_duplicates=False, return_offsets=True,
        )
        rect = None
        bounds = [0, *self.stroke.chunks]
        for a, b in zip(bounds, [*self.stroke.chunks, None], strict=True):
            end = len(samples) if b is None else offsets[b] + 1
            rect = _union(rect, self.polyline(color_img, inv_img, samples[offsets[a]:end]))
        return rect


def render_stroke(color_img: np.ndarray, inv_img: np.ndarray, stroke: Stroke, scale: float = 1.0):
    """Rasterise a finished ``stroke`` into the layers (in place); returns its rect."""
    return _Painter(stroke, scale).render_full(color_img, inv_img)


class Canvas:
    FREEZE_LAG = 3  # segments within this many points of the pen tip may still change

    def __init__(self, width: int, height: int, min_point_dist: float = 1.5, scale: float = 1.0):
        self.width = int(width)
        self.height = int(height)
        self.scale = float(scale)
        self.min_point_dist = float(min_point_dist)
        shape = (self.height, self.width, 3)
        self._color = np.zeros(shape, np.uint8)
        self._inv = np.full(shape, 255, np.uint8)
        self._view_color = np.zeros(shape, np.uint8)
        self._view_inv = np.full(shape, 255, np.uint8)
        self.strokes: list[Stroke] = []
        self.active: Stroke | None = None
        self._painter: _Painter | None = None
        self._frozen = 0           # raw segments of the active stroke already in the committed layers
        self._tail_rect: Rect | None = None
        self._drawn_n = 0          # active point count the view was last built for
        self._active_rect: Rect | None = None
        self._ink_rect: Rect | None = None  # bounding box of all committed pen ink
        self._undo: list[tuple[str, object]] = []
        self._redo: list[tuple[str, object]] = []

    # ------------------------------------------------------------------ state
    @property
    def is_empty(self) -> bool:
        return not self.strokes and (self.active is None or not self.active.points)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def _clip(self, r: Rect | None) -> Rect | None:
        if r is None:
            return None
        x0, y0 = max(0, r[0]), max(0, r[1])
        x1, y1 = min(self.width, r[2]), min(self.height, r[3])
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    # --------------------------------------------------------------- drawing
    def begin(self, color, size: int, erase: bool = False) -> Stroke:
        self.end()
        self.active = Stroke(tuple(int(c) for c in color), int(size), bool(erase))
        self._painter = _Painter(self.active, self.scale)
        self._frozen = 0
        self._tail_rect = None
        self._drawn_n = 0
        self._active_rect = None
        np.copyto(self._view_color, self._color)
        np.copyto(self._view_inv, self._inv)
        return self.active

    def add_point(self, x: float, y: float) -> bool:
        if self.active is None:
            raise RuntimeError("add_point() called without begin()")
        return self.active.add(x, y, self.min_point_dist)

    def _restore_view(self, r: Rect | None) -> None:
        r = self._clip(r)
        if r is not None:
            x0, y0, x1, y1 = r
            self._view_color[y0:y1, x0:x1] = self._color[y0:y1, x0:x1]
            self._view_inv[y0:y1, x0:x1] = self._inv[y0:y1, x0:x1]

    def _sync_active(self) -> None:
        """Freeze settled segments into the committed layers, redraw the tail."""
        a, p = self.active, self._painter
        n = len(a.points)
        if n == self._drawn_n:
            return
        pts = a.points
        dirty = self._tail_rect
        settled = n - 1 - self.FREEZE_LAG  # segments [0, settled) can no longer change
        if n >= 3 and settled > self._frozen:
            r = p.draw_range(self._color, self._inv, pts, self._frozen, settled)
            a.chunks.append(settled)
            self._frozen = settled
            self._active_rect = _union(self._active_rect, r)
            dirty = _union(dirty, r)
        self._restore_view(dirty)
        seg_from = self._frozen if n >= 3 else 0
        self._tail_rect = p.draw_range(self._view_color, self._view_inv, pts, seg_from, None)
        if not a.erase:
            self._active_rect = _union(self._active_rect, self._tail_rect)
        self._drawn_n = n

    def end(self) -> Stroke | None:
        """Finish the active stroke and commit it (undoable). Returns it, if kept."""
        stroke, painter = self.active, self._painter
        self.active, self._painter = None, None
        if stroke is None:
            return None
        if not stroke.points or (stroke.erase and not self.strokes):
            # Nothing to keep (an eraser on an empty canvas changes nothing).
            if self._frozen:
                self._rerender()
            return None
        seg_from = self._frozen if len(stroke.points) >= 3 else 0
        r = painter.draw_range(self._color, self._inv, stroke.points, seg_from, None)
        if not stroke.erase:
            self._ink_rect = _union(self._ink_rect, _union(self._active_rect, r))
        self.strokes.append(stroke)
        self._push(("add", stroke))
        return stroke

    def cancel(self) -> None:
        """Drop the active stroke without committing it."""
        had_frozen = self.active is not None and self._frozen > 0
        self.active, self._painter = None, None
        if had_frozen:
            self._rerender()

    # ---------------------------------------------------------------- history
    def clear(self) -> bool:
        self.cancel()
        if not self.strokes:
            return False
        self._push(("clear", self.strokes))
        self.strokes = []
        self._rerender()
        return True

    def undo(self) -> bool:
        self.end()
        if not self._undo:
            return False
        op, payload = self._undo.pop()
        if op == "add":
            removed = self.strokes.pop()
            assert removed is payload, "history out of sync"
        else:  # "clear"
            self.strokes = list(payload)  # type: ignore[arg-type]
        self._redo.append((op, payload))
        self._rerender()
        return True

    def redo(self) -> bool:
        self.end()
        if not self._redo:
            return False
        op, payload = self._redo.pop()
        if op == "add":
            self.strokes.append(payload)  # type: ignore[arg-type]
            r = render_stroke(self._color, self._inv, payload, self.scale)  # type: ignore[arg-type]
            if not payload.erase:  # type: ignore[union-attr]
                self._ink_rect = _union(self._ink_rect, r)
        else:
            self.strokes = []
            self._rerender()
        self._undo.append((op, payload))
        return True

    def _push(self, entry: tuple[str, object]) -> None:
        self._undo.append(entry)
        self._redo.clear()

    def _rerender(self) -> None:
        self._color[:] = 0
        self._inv[:] = 255
        self._ink_rect = None
        for s in self.strokes:
            r = render_stroke(self._color, self._inv, s, self.scale)
            if not s.erase:
                self._ink_rect = _union(self._ink_rect, r)

    # ------------------------------------------------------------- rendering
    def layers(self) -> tuple[np.ndarray, np.ndarray]:
        """(color, inv) including the active stroke. Treat as read-only."""
        if self.active is None or not self.active.points:
            return self._color, self._inv
        self._sync_active()
        return self._view_color, self._view_inv

    @property
    def ink_rect(self) -> Rect | None:
        """Bounding box that contains every inked pixel (conservative)."""
        r = self._ink_rect
        if self.active is not None and self.active.points and not self.active.erase:
            self.layers()
            r = _union(r, self._active_rect)
        return self._clip(r)

    def composite(self, background: np.ndarray, dim: float = 1.0) -> np.ndarray:
        """Return a new image: ink over ``background`` (optionally dimmed)."""
        if background.shape != self._color.shape:
            raise ValueError(f"background shape {background.shape} != canvas {self._color.shape}")
        dim = float(min(1.0, max(0.0, dim)))
        out = background.copy() if dim >= 1.0 else cv2.convertScaleAbs(background, alpha=dim)
        color, inv = self.layers()
        r = self.ink_rect
        if r is None:
            return out
        x0, y0, x1, y1 = r
        roi = cv2.multiply(background[y0:y1, x0:x1], inv[y0:y1, x0:x1], scale=dim / 255.0)
        out[y0:y1, x0:x1] = cv2.add(roi, color[y0:y1, x0:x1])
        return out

    # ---------------------------------------------------------------- export
    def alpha(self) -> np.ndarray:
        """Committed ink coverage (H x W, uint8)."""
        return 255 - self._inv[:, :, 0]

    def coverage(self) -> np.ndarray:
        """Ink coverage including the live stroke (H x W, uint8)."""
        return 255 - self.layers()[1][:, :, 0]

    def to_bgra(self) -> np.ndarray:
        """Un-premultiplied BGRA image of the committed ink (transparent bg)."""
        a = self.alpha()
        out = np.zeros((self.height, self.width, 4), np.uint8)
        nz = a > 0
        if nz.any():
            af = a[nz].astype(np.float32)[:, None]
            rgb = self._color[nz].astype(np.float32) * 255.0 / af + 0.5
            out[nz, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        out[:, :, 3] = a
        return out

    def to_board(self, bg_color) -> np.ndarray:
        bg = np.empty_like(self._color)
        bg[:] = tuple(int(c) for c in bg_color)
        return cv2.add(cv2.multiply(bg, self._inv, scale=1 / 255.0), self._color)

    def save(self, directory: str | Path, bg_color) -> tuple[Path, Path]:
        """Write ``<stamp>.png`` (on the board colour) and ``<stamp>_transparent.png``."""
        self.end()
        d = Path(directory).expanduser().resolve()
        d.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        board = d / f"airpad_{stamp}.png"
        n = 1
        while board.exists():
            n += 1
            board = d / f"airpad_{stamp}_{n}.png"
        transparent = board.with_name(board.stem + "_transparent.png")
        _imwrite(board, self.to_board(bg_color))
        _imwrite(transparent, self.to_bgra())
        return board, transparent


def _imwrite(path: Path, img: np.ndarray) -> None:
    # cv2.imwrite cannot handle non-ASCII paths on Windows; encode + write instead.
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise OSError(f"PNG encoding failed for {path}")
    path.write_bytes(buf.tobytes())
