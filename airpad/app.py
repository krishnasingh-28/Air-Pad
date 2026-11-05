"""Desktop app: camera/video source + OpenCV window around the engine.

With a window, hand tracking runs on a worker thread while the main thread
renders and displays the previous result (MediaPipe releases the GIL for much
of its work), so throughput is max(inference, render) rather than their sum.
Headless runs (benchmarks, tests) process every frame sequentially instead.
"""

from __future__ import annotations

import contextlib
import json
import sys
import threading
import time

import cv2

from .camera import CameraError, ThreadedCamera, VideoFileSource
from .config import Config, parse_args
from .engine import AirPadEngine, TrackResult
from .model import ensure_model

WINDOW = "AirPad - finger scratch pad"


def _enable_dpi_awareness() -> None:
    """Render 1:1 on high-DPI Windows screens instead of being blurrily upscaled."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
    except Exception:
        with contextlib.suppress(Exception):
            ctypes.windll.user32.SetProcessDPIAware()


def _screen_size() -> tuple[int, int] | None:
    if sys.platform != "win32":
        return None
    import ctypes

    try:
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return None


def _fit_window(size: tuple[int, int]) -> tuple[int, int]:
    w, h = size
    screen = _screen_size()
    if not screen:
        return w, h
    k = min(0.92 * screen[0] / w, 0.85 * screen[1] / h)
    return max(320, int(w * k)), max(180, int(h * k))


def _window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


class TrackingPipeline:
    """Worker thread: read newest frame -> engine.track() -> publish newest result."""

    def __init__(self, source, engine: AirPadEngine, realtime: bool):
        self.source = source
        self.engine = engine
        self.realtime = realtime  # pace video files at their native frame rate
        self._cond = threading.Condition()
        self._result: TrackResult | None = None
        self._seq = 0
        self._running = True
        self.done = False
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._loop, name="tracking", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            seq = 0
            wall0 = time.perf_counter()
            media0: float | None = None
            while self._running:
                item = self.source.read(seq, timeout=0.5)
                if item is None:
                    if self.source.alive:
                        continue  # camera hiccup: keep waiting
                    break  # end of file / camera gone
                frame, ts, seq = item
                if self.realtime:
                    media0 = ts if media0 is None else media0
                    delay = (ts - media0) - (time.perf_counter() - wall0)
                    if delay > 0:
                        time.sleep(delay)
                result = self.engine.track(frame, ts)
                with self._cond:
                    self._result = result
                    self._seq += 1
                    self._cond.notify_all()
        except BaseException as e:  # surfaced to the main thread by get()
            self.error = e
        finally:
            with self._cond:
                self.done = True
                self._cond.notify_all()

    def get(self, last_seq: int, timeout: float):
        """Newest result after ``last_seq`` as ``(result, seq)``; None on timeout/end."""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._seq <= last_seq and not self.done:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            if self.error is not None:
                raise RuntimeError("hand tracking thread failed") from self.error
            if self._seq <= last_seq:
                return None
            return self._result, self._seq

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=5.0)


def _run_headless(cfg: Config, source, engine: AirPadEngine, t0: float) -> None:
    seq = 0
    missed = 0
    while True:
        item = source.read(seq, timeout=2.0)
        if item is None:
            if source.alive and missed < 3:
                missed += 1
                continue
            break
        missed = 0
        frame, ts, seq = item
        engine.process(frame, ts)
        if cfg.max_frames and engine.stats.frames >= cfg.max_frames:
            break
        if cfg.run_seconds and time.perf_counter() - t0 >= cfg.run_seconds:
            break


def _run_window(cfg: Config, source, engine: AirPadEngine, t0: float, is_camera: bool) -> None:
    _enable_dpi_awareness()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, *_fit_window(engine.size))
    cv2.setMouseCallback(WINDOW, lambda ev, x, y, flags, param: engine.handle_mouse(ev, x, y))
    print("[airpad] index finger = draw | two fingers = move/select | open palm = erase | "
          "H = help | Q/Esc = quit")
    pipe = TrackingPipeline(source, engine, realtime=not is_camera)
    fullscreen = False
    seq = 0
    waits = 0
    try:
        while True:
            item = pipe.get(seq, timeout=2.0)
            if item is None:
                if pipe.done:
                    break
                waits += 1
                print("[airpad] waiting for camera frames...")
                if waits >= 3:
                    print("[airpad] camera stopped delivering frames", file=sys.stderr)
                    break
                continue
            waits = 0
            result, seq = item
            out = engine.step(result)
            cv2.imshow(WINDOW, out)
            if is_camera:
                engine.stats.latency_ms.append((time.perf_counter() - result.t) * 1000.0)
            cmd = engine.handle_key(cv2.waitKey(1))
            if cmd == "quit" or _window_closed():
                break
            if cmd == "fullscreen":
                fullscreen = not fullscreen
                cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if cfg.max_frames and engine.stats.frames >= cfg.max_frames:
                break
            if cfg.run_seconds and time.perf_counter() - t0 >= cfg.run_seconds:
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


def run(cfg: Config) -> dict:
    ensure_model(cfg.model_path)
    if cfg.source:
        source = VideoFileSource(cfg.source)
    else:
        source = ThreadedCamera(cfg.camera, cfg.width, cfg.height, cfg.fps, cfg.backend)
    engine = None
    try:
        print("[airpad] loading hand tracking models...")
        engine = AirPadEngine(cfg, source.frame_size)
        (fw, fh), (dw, dh) = source.frame_size, engine.size
        print(f"[airpad] source: {source.backend_name} {fw}x{fh} -> display {dw}x{dh}")
        t0 = time.perf_counter()
        if cfg.headless:
            _run_headless(cfg, source, engine, t0)
        else:
            _run_window(cfg, source, engine, t0, is_camera=not cfg.source)
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        if engine is not None:
            engine.close()
    summary = engine.stats.summary() if engine is not None else {}
    print("[airpad] performance: " + json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    try:
        run(cfg)
    except CameraError as e:
        print(f"[airpad] {e}\n  - is another app using the camera?\n  - try --camera 1 or --backend dshow",
              file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"[airpad] {e}", file=sys.stderr)
        return 2
    return 0
