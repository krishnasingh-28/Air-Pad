"""Frame sources: a low-latency threaded webcam reader and a video-file reader."""

from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np

_BACKENDS = {
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "any": cv2.CAP_ANY,
}


def _backend_order(name: str) -> list[tuple[str, int]]:
    if name != "auto":
        return [(name, _BACKENDS[name])]
    if sys.platform == "win32":
        # MSMF (hw transforms off, see airpad/__init__.py) gives 30 FPS @ 720p;
        # DirectShow is the fallback for cameras MSMF cannot open.
        return [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW), ("any", cv2.CAP_ANY)]
    return [("any", cv2.CAP_ANY)]


class CameraError(RuntimeError):
    pass


class ThreadedCamera:
    """Reads the webcam on a background thread and keeps only the newest frame.

    The main loop never waits on USB / driver buffering and never processes a
    stale frame: if processing is momentarily slower than the camera, older
    frames are dropped instead of queueing up (which would add latency).
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, backend: str = "auto"):
        self.backend_name = ""
        self._cap = None
        errors = []
        for name, api in _backend_order(backend):
            cap = cv2.VideoCapture(index, api)
            if not cap.isOpened():
                cap.release()
                errors.append(f"{name}: not opened")
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                errors.append(f"{name}: opened but returned no frames")
                continue
            self._cap, self.backend_name = cap, name
            break
        if self._cap is None:
            raise CameraError(f"could not open camera {index} ({'; '.join(errors)})")
        self.frame_size = (frame.shape[1], frame.shape[0])
        self._lock = threading.Condition()
        self._frame: np.ndarray | None = frame
        self._ts = time.perf_counter()
        self._seq = 1
        self._running = True
        self._failures = 0
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            ts = time.perf_counter()
            if not ok or frame is None:
                self._failures += 1
                if self._failures > 50:  # camera unplugged / taken by another app
                    break
                time.sleep(0.01)
                continue
            self._failures = 0
            with self._lock:
                self._frame, self._ts = frame, ts
                self._seq += 1
                self._lock.notify_all()
        with self._lock:
            self._running = False
            self._lock.notify_all()

    @property
    def alive(self) -> bool:
        return self._running

    def read(self, last_seq: int = 0, timeout: float = 2.0):
        """Block until a frame newer than ``last_seq`` exists.

        Returns ``(frame, timestamp_seconds, seq)`` or ``None`` on timeout /
        camera failure.
        """
        deadline = time.perf_counter() + timeout
        with self._lock:
            while self._seq <= last_seq and self._running:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self._lock.wait(remaining)
            if self._seq <= last_seq:
                return None
            return self._frame, self._ts, self._seq

    def release(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoFileSource:
    """Plays a video file frame by frame (deterministic; for demos and tests)."""

    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise CameraError(f"could not open video file {path!r}")
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 30.0
        self.backend_name = "file"
        self._seq = 0
        self._eof = False
        ok, frame = self._cap.read()
        if not ok:
            raise CameraError(f"video file {path!r} has no frames")
        self._first = frame
        self.frame_size = (frame.shape[1], frame.shape[0])

    @property
    def alive(self) -> bool:
        return self._cap is not None and not self._eof

    def read(self, last_seq: int = 0, timeout: float = 0.0):
        if not self.alive:
            return None
        if self._first is not None:
            frame, self._first = self._first, None
        else:
            ok, frame = self._cap.read()
            if not ok:
                self._eof = True
                return None
        self._seq += 1
        return frame, self._seq / self.fps, self._seq

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
