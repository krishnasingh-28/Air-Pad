"""On-screen UI drawn with OpenCV: toolbar, custom colour picker, cursor, HUD.

Everything is laid out for a 1280 px wide display and scaled linearly, so the
UI looks the same at 720p, 1080p, or an upscaled 480p camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

PANEL_BG = (38, 34, 32)
BUTTON_BG = (64, 58, 54)
BUTTON_ACTIVE = (200, 140, 40)    # accent (blue-ish, BGR)
TEXT = (240, 240, 240)
TEXT_DIM = (130, 130, 130)
HIGHLIGHT = (255, 255, 255)

BASE_WIDTH = 1280


def bgr_to_hex(c) -> str:
    b, g, r = (int(v) for v in c)
    return f"#{r:02X}{g:02X}{b:02X}"


# ----------------------------------------------------------------- primitives
def text_width(text: str, scale: float, thickness: int = 1, font: int = FONT) -> int:
    return cv2.getTextSize(text, font, scale, thickness)[0][0]


def put_text(img, text, org, scale, color=TEXT, thickness=1, font=FONT, shadow=True):
    x, y = int(org[0]), int(org[1])
    if shadow:
        cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness, AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, AA)


def put_text_centered(img, text, center, scale, color=TEXT, thickness=1, font=FONT, shadow=False):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    put_text(img, text, (center[0] - tw // 2, center[1] + th // 2), scale, color, thickness, font, shadow)


def rounded_rect(img, rect, radius, color, thickness=-1):
    x, y, w, h = (int(v) for v in rect)
    if w <= 0 or h <= 0:
        return
    r = int(max(0, min(radius, w // 2, h // 2)))
    x2, y2 = x + w - 1, y + h - 1
    if thickness < 0:
        if r == 0:
            cv2.rectangle(img, (x, y), (x2, y2), color, -1)
            return
        cv2.rectangle(img, (x + r, y), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x, y + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x + r, y + r), (x2 - r, y + r), (x + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, AA)
        return
    t = int(thickness)
    cv2.line(img, (x + r, y), (x2 - r, y), color, t, AA)
    cv2.line(img, (x + r, y2), (x2 - r, y2), color, t, AA)
    cv2.line(img, (x, y + r), (x, y2 - r), color, t, AA)
    cv2.line(img, (x2, y + r), (x2, y2 - r), color, t, AA)
    if r > 0:
        cv2.ellipse(img, (x + r, y + r), (r, r), 0, 180, 270, color, t, AA)
        cv2.ellipse(img, (x2 - r, y + r), (r, r), 0, 270, 360, color, t, AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, t, AA)
        cv2.ellipse(img, (x + r, y2 - r), (r, r), 0, 90, 180, color, t, AA)


def blend_rect(img, rect, color, opacity):
    """Alpha-fill a rectangle in place (clipped to the image)."""
    x, y, w, h = (int(v) for v in rect)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    solid = np.empty_like(roi)
    solid[:] = color
    cv2.addWeighted(roi, 1.0 - opacity, solid, opacity, 0, dst=roi)


def blend_rounded_rect(img, rect, radius, color, opacity):
    """Alpha-fill a rounded rectangle in place (pixels outside the corners untouched)."""
    x, y, w, h = (int(v) for v in rect)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()
    rounded_rect(layer, (x - x0, y - y0, w, h), radius, color)
    cv2.addWeighted(roi, 1.0 - opacity, layer, opacity, 0, dst=roi)


def progress_ring(img, center, radius, progress, color, thickness):
    if progress <= 0:
        return
    c = (int(center[0]), int(center[1]))
    r = int(radius)
    cv2.circle(img, c, r, (0, 0, 0), thickness + 2, AA)
    cv2.ellipse(img, c, (r, r), -90, 0, 360 * min(1.0, progress), color, thickness, AA)


def rainbow_ring(img, center, radius, thickness):
    for k in range(12):
        hsv = np.uint8([[[k * 15, 230, 255]]])
        col = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
        cv2.ellipse(img, center, (radius, radius), 0, k * 30 - 90, k * 30 - 60, col, thickness, AA)


# -------------------------------------------------------------------- toolbar
@dataclass(frozen=True)
class Button:
    id: str
    x: int
    y: int
    w: int
    h: int
    kind: str  # swatch | custom | picker | text | size
    label: str = ""
    color: tuple[int, int, int] | None = None

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2


@dataclass(frozen=True)
class ToolbarState:
    color: tuple[int, int, int]
    selected: str                       # "color:<i>" or "custom"
    custom_color: tuple[int, int, int] | None
    tool: str                           # pen | eraser
    size: int                           # current tool size (px)
    board: bool
    can_undo: bool
    can_redo: bool


class Toolbar:
    """The bar along the top of the screen. Pre-rendered and cached per state."""

    BAR_H, BTN_H, MARGIN = 72, 44, 14
    FONT_SCALE = 0.5

    def __init__(self, width: int, palette, board_bg=(0, 0, 0)):
        self.width = int(width)
        self.board_bg = tuple(int(c) for c in board_bg)  # bar colour in board mode (matches the page)
        self.s = self.width / BASE_WIDTH
        self.height = round(self.BAR_H * self.s)
        self.palette = tuple(palette)
        self.buttons: tuple[Button, ...] = tuple(self._layout())
        self._cache_key: ToolbarState | None = None
        self._cache: np.ndarray | None = None

    def _layout(self) -> list[Button]:
        s = self.s
        y = (self.BAR_H - self.BTN_H) / 2
        h = self.BTN_H
        items: list[tuple[str, float, float, str, str, tuple | None]] = []
        x = float(self.MARGIN)

        def add(bid, w, kind, label="", color=None, gap=6.0):
            nonlocal x
            items.append((bid, x, w, kind, label, color))
            x += w + gap

        def text_btn_w(label):
            return text_width(label, self.FONT_SCALE) + 24

        for i, (_, col) in enumerate(self.palette):
            add(f"color:{i}", 44, "swatch", color=col, gap=2)
        x += 4
        add("custom", 44, "custom", gap=2)
        add("picker", 44, "picker", gap=16)
        add("tool:pen", text_btn_w("PEN"), "text", "PEN")
        add("tool:eraser", text_btn_w("ERASER"), "text", "ERASER", gap=16)
        add("size:-", 36, "text", "-", gap=4)
        add("size:show", 50, "size", gap=4)
        add("size:+", 36, "text", "+", gap=16)
        for bid, label in (("undo", "UNDO"), ("redo", "REDO"), ("clear", "CLEAR"),
                           ("save", "SAVE"), ("board", "BOARD")):
            add(bid, text_btn_w(label), "text", label)
        return [
            Button(bid, round(bx * s), round(y * s), round(bw * s), round(h * s), kind, label, color)
            for bid, bx, bw, kind, label, color in items
        ]

    @property
    def natural_width(self) -> int:
        """Rightmost button edge + margin (must be <= width)."""
        return max(b.x + b.w for b in self.buttons) + round(self.MARGIN * self.s)

    def contains(self, x: float, y: float) -> bool:
        return 0 <= y < self.height

    def hit(self, x: float, y: float) -> Button | None:
        for b in self.buttons:
            if b.contains(x, y):
                return b
        return None

    def button(self, bid: str) -> Button:
        for b in self.buttons:
            if b.id == bid:
                return b
        raise KeyError(bid)

    def render(self, st: ToolbarState) -> np.ndarray:
        if st == self._cache_key and self._cache is not None:
            return self._cache
        s = self.s
        img = np.empty((self.height, self.width, 3), np.uint8)
        img[:] = self.board_bg if st.board else PANEL_BG
        separator = (40, 40, 40) if st.board else (70, 64, 60)  # faint edge of the drawing area
        cv2.line(img, (0, self.height - 1), (self.width, self.height - 1), separator, max(1, round(s)))
        fs = self.FONT_SCALE * s
        for b in self.buttons:
            cx, cy = b.center
            if b.kind == "swatch":
                selected = st.tool == "pen" and st.selected == b.id
                self._swatch(img, (cx, cy), b.color, selected)
            elif b.kind == "custom":
                selected = st.tool == "pen" and st.selected == "custom"
                if st.custom_color is None:
                    cv2.circle(img, (cx, cy), round(16 * s), TEXT_DIM, max(1, round(1.5 * s)), AA)
                    put_text_centered(img, "?", (cx, cy), 0.5 * s, TEXT_DIM)
                else:
                    self._swatch(img, (cx, cy), st.custom_color, selected)
            elif b.kind == "picker":
                rainbow_ring(img, (cx, cy), round(16 * s), max(2, round(5 * s)))
                put_text_centered(img, "+", (cx, cy), 0.6 * s, TEXT, max(1, round(1.5 * s)))
            elif b.kind == "size":
                rounded_rect(img, (b.x, b.y, b.w, b.h), round(8 * s), BUTTON_BG)
                r = max(1, min(round(st.size / 2), round(11 * s)))
                c = (cx, b.y + round(17 * s))
                if st.tool == "eraser":
                    cv2.circle(img, c, r, TEXT, max(1, round(s)), AA)
                else:
                    cv2.circle(img, c, r, st.color, -1, AA)
                put_text_centered(img, str(st.size), (cx, b.y + b.h - round(9 * s)), 0.38 * s, TEXT)
            else:
                active = (b.id == f"tool:{st.tool}") or (b.id == "board" and st.board)
                disabled = (b.id == "undo" and not st.can_undo) or (b.id == "redo" and not st.can_redo)
                bg = BUTTON_ACTIVE if active else BUTTON_BG
                rounded_rect(img, (b.x, b.y, b.w, b.h), round(8 * s), bg)
                put_text_centered(img, b.label, b.center, fs, TEXT_DIM if disabled else TEXT,
                                  max(1, round(s)))
        self._cache_key, self._cache = st, img
        return img

    def _swatch(self, img, c, color, selected):
        s = self.s
        r = round(16 * s)
        if selected:
            cv2.circle(img, c, r + round(5 * s), HIGHLIGHT, max(2, round(2 * s)), AA)
        cv2.circle(img, c, r, color, -1, AA)
        cv2.circle(img, c, r, (90, 90, 90), max(1, round(s)), AA)

    def overlay(self, out: np.ndarray, st: ToolbarState, opacity: float = 0.9) -> None:
        bar = self.render(st)
        if st.board:
            out[: self.height] = bar  # solid, same colour as the page: crisp buttons
            return
        roi = out[: self.height]
        cv2.addWeighted(roi, 1.0 - opacity, bar, opacity, 0, dst=roi)

    def highlight(self, out: np.ndarray, bid: str) -> None:
        b = self.button(bid)
        pad = round(3 * self.s)
        rounded_rect(out, (b.x - pad, b.y - pad, b.w + 2 * pad, b.h + 2 * pad),
                     round(10 * self.s), HIGHLIGHT, max(1, round(2 * self.s)))


class DwellSelector:
    """Press a button by hovering over it for ``dwell`` seconds.

    One press per visit (move off and back to press again), except ``repeat``
    buttons (e.g. size +/-, undo) which auto-repeat every ``repeat_interval``
    seconds while hovered.
    """

    def __init__(self, dwell: float, repeat: frozenset[str] = frozenset(), repeat_interval: float = 0.35):
        self.dwell = float(dwell)
        self.repeat = frozenset(repeat)
        self.repeat_interval = float(repeat_interval)
        self.target: str | None = None
        self._t0 = 0.0
        self._fired = False
        self.progress = 0.0

    def reset(self) -> None:
        self.target = None
        self._fired = False
        self.progress = 0.0

    def update(self, target: str | None, t: float) -> str | None:
        """Feed the hovered button id (or None); returns an id when pressed."""
        if target != self.target:
            self.target, self._t0, self._fired = target, t, False
        if target is None or self._fired:
            self.progress = 0.0
            return None
        self.progress = min(1.0, (t - self._t0) / self.dwell)
        if self.progress < 1.0:
            return None
        if target in self.repeat:
            # Restart the timer so the next press comes after repeat_interval.
            self._t0 = t - (self.dwell - self.repeat_interval)
        else:
            self._fired = True
        self.progress = 1.0
        return target


# ------------------------------------------------------------- colour picker
def make_spectrum(w: int, h: int) -> np.ndarray:
    """Hue across x; top half white -> pure hue, bottom half pure hue -> dark."""
    hue = (np.arange(w, dtype=np.float32) + 0.5) * (180.0 / w)
    y = np.arange(h, dtype=np.float32)[:, None] + 0.5
    half = h / 2.0
    sat = np.where(y < half, y / half * 255.0, 255.0)
    val = np.where(y < half, 255.0, 255.0 - (y - half) / half * (255.0 - 20.0))
    hsv = np.empty((h, w, 3), np.uint8)
    hsv[..., 0] = np.clip(np.floor(hue), 0, 179)[None, :]
    hsv[..., 1] = np.clip(np.round(sat), 0, 255)
    hsv[..., 2] = np.clip(np.round(val), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def make_gray_strip(w: int, h: int) -> np.ndarray:
    v = np.round(255.0 * (1.0 - (np.arange(h, dtype=np.float32) + 0.5) / h)).astype(np.uint8)
    return np.repeat(np.repeat(v[:, None, None], w, axis=1), 3, axis=2)


class ColorPicker:
    """Modal panel to choose any ink colour with the fingertip (or mouse).

    Point at a colour: the preview updates live. Hold the finger still for
    ``dwell`` seconds to pick it. Hold over *Cancel* (or press Esc) to close.
    """

    def __init__(self, width: int, height: int, top: int, dwell: float = 0.9):
        s = width / BASE_WIDTH
        self.s = s
        self.dwell = float(dwell)
        self.still_radius = 18.0 * s
        pw = min(round(860 * s), width - round(40 * s))
        ph = min(round(400 * s), height - top - round(20 * s))
        x0 = (width - pw) // 2
        y0 = top + max(round(10 * s), (height - top - ph) // 2)
        pad, gap = round(20 * s), round(14 * s)
        title_h = round(52 * s)
        gray_w, col_w = round(34 * s), round(170 * s)
        sx, sy = x0 + pad, y0 + title_h
        sh = ph - title_h - pad
        sw = pw - 2 * pad - gray_w - col_w - 2 * gap
        if sw < 60 or sh < 60:
            raise ValueError(f"display {width}x{height} too small for the color picker")
        gx = sx + sw + gap
        cx = gx + gray_w + gap
        self.panel = (x0, y0, pw, ph)
        self.spec_rect = (sx, sy, sw, sh)
        self.gray_rect = (gx, sy, gray_w, sh)
        self.preview_rect = (cx, sy + round(22 * s), col_w, round(86 * s))
        self.current_rect = (cx, sy + round(178 * s), col_w, round(40 * s))
        self.cancel_rect = (cx, sy + sh - round(44 * s), col_w, round(44 * s))
        self.col_x = cx
        self._spectrum = make_spectrum(sw, sh)
        self._gray = make_gray_strip(gray_w, sh)
        self.is_open = False
        self.hover_pt: tuple[float, float] | None = None
        self.hover_color: tuple[int, int, int] | None = None
        self.progress = 0.0
        self._anchor: tuple[float, float] | None = None
        self._anchor_t = 0.0
        self._anchor_kind: str | None = None

    @staticmethod
    def _inside(rect, x, y) -> bool:
        rx, ry, rw, rh = rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def open(self) -> None:
        self.is_open = True
        self._reset_hover()

    def close(self) -> None:
        self.is_open = False
        self._reset_hover()

    def _reset_hover(self) -> None:
        self.hover_pt = None
        self.hover_color = None
        self.progress = 0.0
        self._anchor = None
        self._anchor_kind = None

    def target_at(self, x: float, y: float) -> tuple[str | None, tuple[int, int, int] | None]:
        """('color', bgr) over a colour, ('cancel', None) over Cancel, else (None, None)."""
        for rect, img in ((self.spec_rect, self._spectrum), (self.gray_rect, self._gray)):
            if self._inside(rect, x, y):
                px = min(int(x - rect[0]), rect[2] - 1)
                py = min(int(y - rect[1]), rect[3] - 1)
                return "color", tuple(int(v) for v in img[py, px])
        if self._inside(self.cancel_rect, x, y):
            return "cancel", None
        return None, None

    def update(self, pt, t: float):
        """Feed the fingertip (or None). Returns ('pick', bgr) / ('cancel', None) / None."""
        if not self.is_open:
            return None
        if pt is None:
            self._reset_hover()
            return None
        x, y = float(pt[0]), float(pt[1])
        kind, color = self.target_at(x, y)
        self.hover_pt = (x, y)
        self.hover_color = color
        if kind is None:
            self._anchor = None
            self._anchor_kind = None
            self.progress = 0.0
            return None
        if (self._anchor is None or self._anchor_kind != kind
                or math.hypot(x - self._anchor[0], y - self._anchor[1]) > self.still_radius):
            self._anchor, self._anchor_t, self._anchor_kind = (x, y), t, kind
        dwell = self.dwell if kind == "color" else self.dwell * 0.7
        self.progress = min(1.0, (t - self._anchor_t) / dwell)
        if self.progress < 1.0:
            return None
        self.close()
        return ("pick", color) if kind == "color" else ("cancel", None)

    def click(self, x: float, y: float):
        """Mouse click: pick / cancel immediately; clicks elsewhere are ignored."""
        if not self.is_open:
            return None
        kind, color = self.target_at(x, y)
        if kind is None:
            return None
        self.close()
        return ("pick", color) if kind == "color" else ("cancel", None)

    def render(self, out: np.ndarray, current_color) -> None:
        if not self.is_open:
            return
        s = self.s
        x0, y0, pw, ph = self.panel
        blend_rect(out, (0, 0, out.shape[1], out.shape[0]), (0, 0, 0), 0.35)
        rounded_rect(out, self.panel, round(14 * s), PANEL_BG)
        rounded_rect(out, self.panel, round(14 * s), (90, 84, 80), max(1, round(s)))
        put_text(out, "Pick an ink color", (x0 + round(20 * s), y0 + round(34 * s)), 0.75 * s, TEXT,
                 max(1, round(1.5 * s)), FONT_BOLD, shadow=False)
        put_text(out, "point at a color and hold still", (x0 + round(300 * s), y0 + round(33 * s)),
                 0.5 * s, TEXT_DIM, 1, FONT, shadow=False)
        for rect, img in ((self.spec_rect, self._spectrum), (self.gray_rect, self._gray)):
            rx, ry, rw, rh = rect
            out[ry:ry + rh, rx:rx + rw] = img
            cv2.rectangle(out, (rx - 1, ry - 1), (rx + rw, ry + rh), (90, 84, 80), 1)

        shown = self.hover_color if self.hover_color is not None else current_color
        cx = self.col_x
        put_text(out, "PREVIEW",
                 (cx, self.spec_rect[1] + round(14 * s)), 0.45 * s, TEXT_DIM, 1, shadow=False)
        rounded_rect(out, self.preview_rect, round(10 * s), shown)
        rounded_rect(out, self.preview_rect, round(10 * s), (110, 104, 100), max(1, round(s)))
        put_text(out, bgr_to_hex(shown), (cx, self.spec_rect[1] + round(134 * s)), 0.6 * s, TEXT,
                 max(1, round(s)), shadow=False)
        put_text(out, "CURRENT", (cx, self.current_rect[1] - round(8 * s)), 0.45 * s, TEXT_DIM, 1,
                 shadow=False)
        rounded_rect(out, self.current_rect, round(8 * s), tuple(int(v) for v in current_color))
        rounded_rect(out, self.current_rect, round(8 * s), (110, 104, 100), max(1, round(s)))

        over_cancel = self.hover_pt is not None and self._inside(self.cancel_rect, *self.hover_pt)
        rounded_rect(out, self.cancel_rect, round(8 * s), BUTTON_ACTIVE if over_cancel else BUTTON_BG)
        cr = self.cancel_rect
        put_text_centered(out, "CANCEL", (cr[0] + cr[2] // 2, cr[1] + cr[3] // 2), 0.5 * s, TEXT,
                          max(1, round(s)))

        if self.hover_pt is not None and self.hover_color is not None:
            hx, hy = int(self.hover_pt[0]), int(self.hover_pt[1])
            r = round(10 * s)
            cv2.circle(out, (hx, hy), r + 2, (0, 0, 0), max(1, round(2 * s)), AA)
            cv2.circle(out, (hx, hy), r, (255, 255, 255), max(1, round(2 * s)), AA)


# ------------------------------------------------------------------ overlays
def draw_landmarks(img, pts, s, line_color=(230, 230, 230), joint_color=(0, 200, 255),
                   connections=()):
    if pts is None:
        return
    p = np.round(pts).astype(np.int32)
    lines = [np.array([p[a], p[b]], np.int32) for a, b in connections]
    cv2.polylines(img, lines, False, line_color, max(1, round(1.5 * s)), AA)
    r = max(2, round(3 * s))
    for x, y in p:
        cv2.circle(img, (int(x), int(y)), r, joint_color, -1, AA)


def draw_badge(img, text, org, s, bg, fg=TEXT) -> int:
    """Pill with text at bottom-left ``org``; returns its width."""
    fs = 0.55 * s
    tw = text_width(text, fs, max(1, round(s)), FONT_BOLD)
    h = round(30 * s)
    w = tw + round(24 * s)
    x, y = int(org[0]), int(org[1]) - h
    rounded_rect(img, (x, y, w, h), h // 2, bg)
    put_text(img, text, (x + round(12 * s), y + h - round(9 * s)), fs, fg, max(1, round(s)), FONT_BOLD,
             shadow=False)
    return w


def draw_toast(img, text, top, s, opacity=1.0):
    if opacity <= 0:
        return
    fs = 0.6 * s
    tw = text_width(text, fs, max(1, round(s)))
    h = round(38 * s)
    w = min(tw + round(36 * s), img.shape[1])
    x = max(0, (img.shape[1] - w) // 2)
    y = top + round(14 * s)
    roi = img[y:y + h, x:x + w]
    if roi.size == 0:
        return
    layer = roi.copy()
    rounded_rect(layer, (0, 0, roi.shape[1], roi.shape[0]), roi.shape[0] // 2, (20, 20, 20))
    put_text(layer, text, (round(18 * s), roi.shape[0] - round(13 * s)), fs, TEXT, max(1, round(s)),
             shadow=False)
    cv2.addWeighted(roi, 1.0 - opacity * 0.92, layer, opacity * 0.92, 0, dst=roi)


HELP_LINES = (
    ("GESTURES", None),
    ("Index finger up", "draw"),
    ("Index + middle (peace sign)", "move without drawing / select"),
    ("Open palm", "erase under your palm"),
    ("Point at a toolbar button & hold", "press it"),
    ("Rainbow + button", "pick any custom ink color"),
    ("KEYS", None),
    ("1-9  /  P", "preset colors  /  color picker"),
    ("E  /  [ ]", "eraser tool  /  brush size"),
    ("Z  Y  C", "undo  redo  clear"),
    ("S", "save PNG (board + transparent)"),
    ("B  L  F  M", "board  landmarks  fullscreen  mirror"),
    ("H  /  Q or Esc", "help  /  quit"),
)


def draw_help(img, top, s):
    w, h = round(700 * s), round(470 * s)
    x = (img.shape[1] - w) // 2
    y = top + max(round(10 * s), (img.shape[0] - top - h) // 2)
    blend_rounded_rect(img, (x, y, w, h), round(14 * s), (20, 18, 16), 0.88)
    rounded_rect(img, (x, y, w, h), round(14 * s), (90, 84, 80), max(1, round(s)))
    yy = y + round(40 * s)
    for left, right in HELP_LINES:
        if right is None:
            put_text(img, left, (x + round(28 * s), yy), 0.6 * s, (0, 200, 255), max(1, round(s)),
                     FONT_BOLD, shadow=False)
        else:
            put_text(img, left, (x + round(40 * s), yy), 0.52 * s, TEXT, 1, shadow=False)
            put_text(img, right, (x + round(370 * s), yy), 0.52 * s, TEXT_DIM, 1, shadow=False)
        yy += round(33 * s)
