"""The AirPad engine: one call per frame, no window or camera dependencies.

    frame -> hand landmarks -> finger states -> debounced gesture
          -> One Euro smoothed fingertip -> toolbar / picker / stroke logic
          -> canvas -> composited output frame

Keeping this independent of the GUI makes the whole pipeline testable with
recorded or synthetic frames and a scripted fake tracker.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .canvas import Canvas
from .config import ERASER_SIZES, PALETTE, PEN_SIZES, Config
from .filters import OneEuroFilter
from .gestures import (
    HAND_CONNECTIONS,
    INDEX_TIP,
    PALM_POINTS,
    FingerStateEstimator,
    Gesture,
    GestureStabilizer,
    classify,
)
from .tracker import HandObservation
from .ui import (
    BASE_WIDTH,
    ColorPicker,
    DwellSelector,
    Toolbar,
    ToolbarState,
    bgr_to_hex,
    draw_badge,
    draw_help,
    draw_landmarks,
    draw_toast,
    progress_ring,
    put_text,
)


@dataclass(frozen=True)
class TrackResult:
    frame: np.ndarray       # (mirrored) camera frame the landmarks refer to
    obs: HandObservation | None
    t: float                # capture time, seconds
    infer_ms: float
    mirrored: bool


MODE_STYLE = {
    Gesture.DRAW: ("DRAW", (60, 170, 60)),
    Gesture.HOVER: ("MOVE", (200, 140, 40)),
    Gesture.ERASE: ("ERASE", (40, 120, 230)),
    Gesture.IDLE: ("IDLE", (90, 90, 90)),
    Gesture.NONE: ("NO HAND", (60, 60, 150)),
}


class Stats:
    """Rolling performance numbers (milliseconds / frames per second)."""

    def __init__(self, window: int = 60, keep: int = 20000):
        self._ticks: deque[float] = deque(maxlen=window)
        self.infer_ms: deque[float] = deque(maxlen=keep)
        self.render_ms: deque[float] = deque(maxlen=keep)
        self.latency_ms: deque[float] = deque(maxlen=keep)
        self.frames = 0

    def add(self, infer_ms: float, render_ms: float) -> None:
        self._ticks.append(time.perf_counter())
        self.infer_ms.append(infer_ms)
        self.render_ms.append(render_ms)
        self.frames += 1

    @property
    def fps(self) -> float:
        if len(self._ticks) < 2:
            return 0.0
        span = self._ticks[-1] - self._ticks[0]
        return (len(self._ticks) - 1) / span if span > 0 else 0.0

    @staticmethod
    def _recent_mean(d: deque, n: int = 30) -> float:
        if not d:
            return 0.0
        k = min(n, len(d))
        return sum(list(d)[-k:]) / k

    @property
    def infer_recent(self) -> float:
        return self._recent_mean(self.infer_ms)

    def summary(self) -> dict:
        def pct(d, q):
            return round(float(np.percentile(np.asarray(d), q)), 2) if d else None

        return {
            "frames": self.frames,
            "fps": round(self.fps, 1),
            "hand_ms_p50": pct(self.infer_ms, 50),
            "hand_ms_p95": pct(self.infer_ms, 95),
            "render_ms_p50": pct(self.render_ms, 50),
            "render_ms_p95": pct(self.render_ms, 95),
            "latency_ms_p50": pct(self.latency_ms, 50),
            "latency_ms_p95": pct(self.latency_ms, 95),
        }


class AirPadEngine:
    def __init__(self, cfg: Config, frame_size: tuple[int, int], tracker=None):
        self.cfg = cfg
        fw, fh = int(frame_size[0]), int(frame_size[1])
        if fw <= 0 or fh <= 0:
            raise ValueError(f"invalid frame size {frame_size}")
        if fw < cfg.min_display_width:
            k = cfg.min_display_width / fw
            self.size = (cfg.min_display_width, int(round(fh * k)))
        else:
            self.size = (fw, fh)
        W, H = self.size
        self.s = W / BASE_WIDTH

        if tracker is None:
            from .tracker import HandTracker

            tracker = HandTracker(
                cfg.model_path,
                num_hands=1,
                min_detection_confidence=cfg.min_detection_confidence,
                min_presence_confidence=cfg.min_presence_confidence,
                min_tracking_confidence=cfg.min_tracking_confidence,
                infer_width=cfg.infer_width,
                frame_size=(fw, fh),  # build tile trackers now, not on the first frame
            )
            self._owns_tracker = True
        else:
            self._owns_tracker = False
        self.tracker = tracker

        self.canvas = Canvas(W, H, min_point_dist=cfg.min_point_dist * self.s, scale=self.s)
        self.fingers = FingerStateEstimator()
        self.stabilizer = GestureStabilizer(cfg.confirm_frames)
        self.tip_filter = OneEuroFilter(cfg.filter_min_cutoff, cfg.filter_beta, cfg.filter_d_cutoff)
        self.palm_filter = OneEuroFilter(cfg.filter_min_cutoff, cfg.filter_beta, cfg.filter_d_cutoff)
        self.toolbar = Toolbar(W, PALETTE, board_bg=cfg.board_color)
        self.picker = ColorPicker(W, H, top=self.toolbar.height, dwell=cfg.picker_dwell_seconds)
        self.dwell = DwellSelector(cfg.dwell_seconds, repeat=frozenset({"size:-", "size:+", "undo", "redo"}))
        self.jump_px = cfg.jump_fraction * W

        # Ink state
        self.color: tuple[int, int, int] = PALETTE[0][1]
        self.color_id = "color:0"
        self.custom_color: tuple[int, int, int] | None = None
        self.tool = "pen"
        self.pen_idx = int(np.clip(cfg.pen_size_index, 0, len(PEN_SIZES) - 1))
        self.eraser_idx = int(np.clip(cfg.eraser_size_index, 0, len(ERASER_SIZES) - 1))

        # View state
        self.board = cfg.start_in_board
        self.show_landmarks = cfg.show_landmarks
        self.show_help = False
        self.mirror = cfg.mirror
        self._board_bg = np.empty((H, W, 3), np.uint8)
        self._board_bg[:] = cfg.board_color

        # Tracking / interaction state
        self.gesture = Gesture.NONE
        self.raw_gesture = Gesture.NONE
        self.tip: np.ndarray | None = None
        self.palm: np.ndarray | None = None
        self.landmarks_px: np.ndarray | None = None
        self.hover_button: str | None = None
        self._lost = 0
        self._active_mode: str | None = None  # pen | eraser | palm while a stroke is open
        self._release_required = False
        self._toast_text = ""
        self._toast_until = 0.0
        self._t_start: float | None = None
        self._now = 0.0
        self.stats = Stats()

    # ------------------------------------------------------------ properties
    @property
    def pen_size(self) -> int:
        return max(1, round(PEN_SIZES[self.pen_idx] * self.s))

    @property
    def eraser_size(self) -> int:
        return max(2, round(ERASER_SIZES[self.eraser_idx] * self.s))

    @property
    def palm_eraser_size(self) -> int:
        return max(2, round(self.cfg.palm_eraser_size * self.s))

    @property
    def drawing(self) -> bool:
        return self.canvas.active is not None

    # ----------------------------------------------------------------- frame
    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Process one BGR camera frame captured at time ``t`` (seconds)."""
        return self.step(self.track(frame, t))

    def track(self, frame: np.ndarray, t: float) -> TrackResult:
        """Stage 1 - hand tracking. Safe to run on a worker thread: it only
        touches the tracker (and reads ``mirror``), never interaction state."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("expected a BGR frame of shape (h, w, 3)")
        t0 = time.perf_counter()
        mirrored = self.mirror
        if mirrored:
            frame = cv2.flip(frame, 1)
        obs = self.tracker.process(frame, int(round(t * 1000.0)))
        return TrackResult(frame, obs, t, (time.perf_counter() - t0) * 1000.0, mirrored)

    def step(self, tr: TrackResult) -> np.ndarray:
        """Stage 2 - gestures, strokes, UI and rendering (main thread)."""
        t1 = time.perf_counter()
        t = tr.t
        if self._t_start is None:
            self._t_start = t
        self._now = t
        frame, obs = tr.frame, tr.obs
        if tr.mirrored != self.mirror:
            # Mirroring was toggled while this frame was in flight.
            frame, obs = cv2.flip(frame, 1), None
        W, H = self.size
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        self._interact(obs, t)
        out = self._render(frame, t)
        self.stats.add(tr.infer_ms, (time.perf_counter() - t1) * 1000.0)
        return out

    # ----------------------------------------------------------- interaction
    def _reset_tracking(self) -> None:
        self.fingers.reset()
        self.stabilizer.reset()
        self.tip_filter.reset()
        self.palm_filter.reset()
        self.dwell.reset()
        self.gesture = Gesture.NONE
        self.raw_gesture = Gesture.NONE
        self.tip = None
        self.palm = None
        self.hover_button = None
        self._release_required = False

    def _end_stroke(self) -> None:
        self.canvas.end()
        self._active_mode = None

    def _interact(self, obs, t: float) -> None:
        W, H = self.size
        if obs is not None and obs.fresh:
            # First landmarks after a palm detection are coarse (can be ~100 px
            # off, wrong gesture); tracking refines them next frame. Treat this
            # frame like a brief drop-out so it can't leave a stray mark.
            obs = None
        if obs is None:
            self._lost += 1
            self.landmarks_px = None
            self.hover_button = None
            self.dwell.update(None, t)
            if self.picker.is_open:
                self.picker.update(None, t)
            if self._lost > self.cfg.lost_grace_frames:
                # Hand really gone: lift the pen and forget the motion history.
                self._end_stroke()
                self._reset_tracking()
            return
        self._lost = 0

        px = obs.norm[:, :2] * np.array([W, H], dtype=np.float64)
        self.landmarks_px = px
        self.raw_gesture = classify(self.fingers.update(obs.world))
        g = self.stabilizer.update(self.raw_gesture)
        self.gesture = g
        tip = self.tip_filter(px[INDEX_TIP], t)
        palm = self.palm_filter(px[list(PALM_POINTS)].mean(axis=0), t)
        self.tip, self.palm = tip, palm
        pointing = g in (Gesture.DRAW, Gesture.HOVER)
        if self._release_required and g is not Gesture.DRAW:
            self._release_required = False

        # 1) Modal colour picker swallows all input.
        if self.picker.is_open:
            self._end_stroke()
            self.hover_button = None
            event = self.picker.update(tip if pointing else None, t)
            if event is not None:
                self._picker_event(event)
            return

        # 2) Toolbar: point at a button and hold to press it.
        if pointing and self.toolbar.contains(*tip):
            self._end_stroke()
            btn = self.toolbar.hit(*tip)
            self.hover_button = btn.id if btn is not None and btn.id != "size:show" else None
            fired = self.dwell.update(self.hover_button, t)
            if fired:
                self.do_action(fired)
            return
        self.hover_button = None
        self.dwell.update(None, t)

        # 3) Strokes.
        if g is Gesture.DRAW and not self._release_required:
            mode = "eraser" if self.tool == "eraser" else "pen"
        elif g is Gesture.ERASE:
            mode = "palm"
        else:
            mode = None
        if mode is None:
            self._end_stroke()
            return
        if self.stabilizer.pending:
            return  # pose is changing: hold the pen still, no tail, no break
        pt = palm if mode == "palm" else tip
        active = self.canvas.active
        if active is not None and self._active_mode != mode:
            self._end_stroke()
            active = None
        if active is not None and active.points:
            lx, ly = active.points[-1]
            if math.hypot(pt[0] - lx, pt[1] - ly) > self.jump_px:
                self._end_stroke()  # tracking glitch / teleport: don't draw a long line
                active = None
        if active is None:
            if mode == "pen":
                self.canvas.begin(self.color, self.pen_size, erase=False)
            elif mode == "eraser":
                self.canvas.begin((0, 0, 0), self.eraser_size, erase=True)
            else:
                self.canvas.begin((0, 0, 0), self.palm_eraser_size, erase=True)
            self._active_mode = mode
        self.canvas.add_point(float(pt[0]), float(pt[1]))

    def _picker_event(self, event) -> None:
        kind, color = event
        if kind == "pick" and color is not None:
            self.custom_color = tuple(int(c) for c in color)
            self.color = self.custom_color
            self.color_id = "custom"
            self.tool = "pen"
            self.toast(f"Ink {bgr_to_hex(self.color)}")
        # Don't start drawing just because the finger is still in DRAW pose
        # where the picker used to be: wait for the pose to be released once.
        self._release_required = True

    # --------------------------------------------------------------- actions
    def toast(self, text: str, seconds: float = 1.6) -> None:
        self._toast_text = text
        self._toast_until = self._now + seconds

    def open_picker(self) -> None:
        self._end_stroke()
        self.dwell.reset()
        self.hover_button = None
        self.picker.open()

    def do_action(self, action: str) -> None:
        if action.startswith("color:"):
            i = int(action.split(":", 1)[1])
            if 0 <= i < len(PALETTE):
                self.color = PALETTE[i][1]
                self.color_id = action
                self.tool = "pen"
        elif action == "custom":
            if self.custom_color is None:
                self.open_picker()
            else:
                self.color = self.custom_color
                self.color_id = "custom"
                self.tool = "pen"
        elif action == "picker":
            self.open_picker()
        elif action == "tool:pen":
            self.tool = "pen"
        elif action == "tool:eraser":
            self.tool = "eraser"
        elif action in ("size:-", "size:+"):
            step = 1 if action == "size:+" else -1
            if self.tool == "eraser":
                self.eraser_idx = int(np.clip(self.eraser_idx + step, 0, len(ERASER_SIZES) - 1))
                self.toast(f"Eraser {self.eraser_size}px", 0.9)
            else:
                self.pen_idx = int(np.clip(self.pen_idx + step, 0, len(PEN_SIZES) - 1))
                self.toast(f"Brush {self.pen_size}px", 0.9)
        elif action == "undo":
            self._active_mode = None
            self.toast("Undo" if self.canvas.undo() else "Nothing to undo", 0.9)
        elif action == "redo":
            self._active_mode = None
            self.toast("Redo" if self.canvas.redo() else "Nothing to redo", 0.9)
        elif action == "clear":
            self._end_stroke()
            self.toast("Cleared  (Z to undo)" if self.canvas.clear() else "Already empty", 1.2)
        elif action == "save":
            self.save()
        elif action == "board":
            self.board = not self.board
            self.toast("Board mode" if self.board else "Camera mode", 0.9)
        elif action == "help":
            self.show_help = not self.show_help
        elif action == "landmarks":
            self.show_landmarks = not self.show_landmarks
        elif action == "mirror":
            self.mirror = not self.mirror
            self._end_stroke()
            self._reset_tracking()
            self.toast("Mirror on" if self.mirror else "Mirror off", 0.9)
        elif action == "toggle_eraser":
            self.tool = "pen" if self.tool == "eraser" else "eraser"

    def save(self) -> tuple[Path, Path] | None:
        self._active_mode = None
        if self.canvas.is_empty:
            self.toast("Nothing to save yet")
            return None
        try:
            board, transparent = self.canvas.save(self.cfg.save_dir, self.cfg.board_color)
        except OSError as e:
            self.toast(f"Save failed: {e.strerror or e}")
            return None
        print(f"[airpad] saved {board}\n[airpad] saved {transparent}")
        self.toast(f"Saved {board.name}", 2.5)
        return board, transparent

    KEYMAP = {
        ord("c"): "clear", ord("z"): "undo", ord("y"): "redo", ord("s"): "save",
        ord("b"): "board", ord("e"): "toggle_eraser", ord("p"): "picker",
        ord("l"): "landmarks", ord("h"): "help", ord("m"): "mirror",
        ord("["): "size:-", ord("-"): "size:-", ord("]"): "size:+", ord("="): "size:+", ord("+"): "size:+",
    }

    def handle_key(self, key: int) -> str | None:
        """Returns 'quit' or 'fullscreen' for the app layer, else None."""
        if key is None or key < 0:
            return None
        k = key & 0xFF
        if k == 27:  # Esc closes overlays first
            if self.picker.is_open:
                self.picker.close()
                self._release_required = True
                return None
            if self.show_help:
                self.show_help = False
                return None
            return "quit"
        if ord("A") <= k <= ord("Z"):
            k += 32  # case-insensitive
        if k == ord("q"):
            return "quit"
        if k == ord("f"):
            return "fullscreen"
        if ord("1") <= k <= ord("9"):
            self.do_action(f"color:{k - ord('1')}")
            return None
        action = self.KEYMAP.get(k)
        if action:
            self.do_action(action)
        return None

    def handle_mouse(self, event: int, x: int, y: int) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.picker.is_open:
            result = self.picker.click(x, y)
            if result is not None:
                self._picker_event(result)
            return
        btn = self.toolbar.hit(x, y)
        if btn is not None and btn.id != "size:show":
            self.do_action(btn.id)

    # -------------------------------------------------------------- rendering
    def _toolbar_state(self) -> ToolbarState:
        return ToolbarState(
            color=self.color,
            selected=self.color_id,
            custom_color=self.custom_color,
            tool=self.tool,
            size=self.eraser_size if self.tool == "eraser" else self.pen_size,
            board=self.board,
            can_undo=self.canvas.can_undo,
            can_redo=self.canvas.can_redo,
        )

    def _render(self, frame: np.ndarray, t: float) -> np.ndarray:
        s = self.s
        if self.board:
            out = self.canvas.composite(self._board_bg)
            self._draw_pip(out, frame)
        else:
            out = self.canvas.composite(frame, self.cfg.dim)
            if self.show_landmarks and self.landmarks_px is not None:
                draw_landmarks(out, self.landmarks_px, s, connections=HAND_CONNECTIONS)
        self.toolbar.overlay(out, self._toolbar_state())
        if self.hover_button is not None:
            self.toolbar.highlight(out, self.hover_button)
        self.picker.render(out, self.color)
        self._draw_cursor(out)
        self._draw_hud(out, t)
        return out

    def _draw_pip(self, out: np.ndarray, frame: np.ndarray) -> None:
        W, H = self.size
        s = self.s
        pw = round(W * 0.22)
        ph = round(pw * H / W)
        m = round(16 * s)
        x, y = W - pw - m, H - ph - m
        small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        if self.show_landmarks and self.landmarks_px is not None:
            draw_landmarks(small, self.landmarks_px * (pw / W), s * 0.5, connections=HAND_CONNECTIONS)
        out[y:y + ph, x:x + pw] = small
        cv2.rectangle(out, (x - 1, y - 1), (x + pw, y + ph), (110, 104, 100), max(1, round(s)))

    def _draw_cursor(self, out: np.ndarray) -> None:
        if self.tip is None:
            return
        s = self.s
        g = self.gesture
        tx, ty = int(round(self.tip[0])), int(round(self.tip[1]))
        th = max(1, round(2 * s))
        ring_r, ring_t = round(24 * s), max(2, round(4 * s))
        if self.picker.is_open:
            if g in (Gesture.DRAW, Gesture.HOVER):
                progress_ring(out, (tx, ty), ring_r, self.picker.progress, (255, 255, 255), ring_t)
            return
        if self.hover_button is not None:
            cv2.circle(out, (tx, ty), round(7 * s), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (tx, ty), round(7 * s), (0, 0, 0), max(1, round(s)), cv2.LINE_AA)
            progress_ring(out, (tx, ty), ring_r, self.dwell.progress, (255, 255, 255), ring_t)
            return
        if g is Gesture.ERASE and self.palm is not None:
            px, py = int(round(self.palm[0])), int(round(self.palm[1]))
            r = self.palm_eraser_size // 2
            cv2.circle(out, (px, py), r, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.circle(out, (px, py), r, (255, 255, 255), th, cv2.LINE_AA)
            return
        if g is Gesture.DRAW and not self._release_required:
            if self.tool == "eraser":
                r = self.eraser_size // 2
                cv2.circle(out, (tx, ty), r, (0, 0, 0), th + 2, cv2.LINE_AA)
                cv2.circle(out, (tx, ty), r, (255, 255, 255), th, cv2.LINE_AA)
            else:
                r = max(2, self.pen_size // 2)
                cv2.circle(out, (tx, ty), r + th, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(out, (tx, ty), r, self.color, -1, cv2.LINE_AA)
            return
        # HOVER / IDLE / waiting for release: hollow ring = pen is up.
        col = self.color if g is Gesture.HOVER else (160, 160, 160)
        r = max(round(9 * s), self.pen_size // 2 + round(4 * s))
        cv2.circle(out, (tx, ty), r, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.circle(out, (tx, ty), r, col, th, cv2.LINE_AA)
        cv2.circle(out, (tx, ty), max(1, round(2 * s)), col, -1, cv2.LINE_AA)

    def _draw_hud(self, out: np.ndarray, t: float) -> None:
        W, H = self.size
        s = self.s
        m = round(16 * s)
        label, col = MODE_STYLE[self.gesture]
        if self.gesture is Gesture.DRAW and self.tool == "eraser":
            label = "ERASER"
        w = draw_badge(out, label, (m, H - m), s, col)
        perf = f"{self.stats.fps:4.1f} FPS   hand {self.stats.infer_recent:4.1f} ms"
        put_text(out, perf, (m + w + round(12 * s), H - m - round(9 * s)), 0.5 * s, (230, 230, 230), 1)
        if self._toast_text and t < self._toast_until:
            remaining = self._toast_until - t
            draw_toast(out, self._toast_text, self.toolbar.height, s, opacity=min(1.0, remaining / 0.3))
        if self.show_help:
            draw_help(out, self.toolbar.height, s)
        elif self._t_start is not None and t - self._t_start < 12.0 and not self.picker.is_open:
            hint = "Index finger: draw  |  Two fingers: move & select  |  Open palm: erase  |  H: help"
            fs = 0.5 * s
            tw = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
            put_text(out, hint, ((W - tw) // 2, H - round(60 * s)), fs, (235, 235, 235), 1)

    def close(self) -> None:
        if self._owns_tracker and self.tracker is not None:
            self.tracker.close()
        self.tracker = None
