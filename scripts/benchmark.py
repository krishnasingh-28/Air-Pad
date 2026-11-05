"""Benchmark AirPad's pipeline without a webcam.

Builds a synthetic 1280x720 video from the sample hand photos in
tests/fixtures (write a line, lift, write again), then measures:

* sequential: hand tracking ms + render ms per frame
* pipelined : real throughput with tracking on a worker thread (as the app runs)

Usage:  python scripts/benchmark.py [--frames N]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import airpad  # noqa: E402,F401  (package init sets OpenCV/MediaPipe env vars)
from airpad.app import TrackingPipeline  # noqa: E402
from airpad.camera import VideoFileSource  # noqa: E402
from airpad.config import Config  # noqa: E402
from airpad.engine import AirPadEngine  # noqa: E402
from airpad.model import ensure_model  # noqa: E402

W, H, PHOTO = 1280, 720, 560


def scenario(n: int) -> list[np.ndarray]:
    fx = ROOT / "tests" / "fixtures"
    point = cv2.resize(cv2.imread(str(fx / "pointing_up.jpg")), (PHOTO, PHOTO))
    peace = cv2.resize(cv2.imread(str(fx / "victory.jpg")), (PHOTO, PHOTO))
    frames = []
    for i in range(n):
        phase = i % 120
        f = np.full((H, W, 3), (120, 130, 140), np.uint8)
        if phase < 90:
            x = int(40 + 600 * (phase / 89 if (i // 120) % 2 == 0 else 1 - phase / 89))
            f[120:120 + PHOTO, x:x + PHOTO] = point
        else:
            f[120:120 + PHOTO, 640:640 + PHOTO] = peace
        frames.append(f)
    return frames


def pct(values, q):
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=240)
    args = ap.parse_args()
    cfg = Config()
    cfg.mirror = False
    ensure_model(cfg.model_path)
    frames = scenario(args.frames)

    # Sequential: exact per-stage cost.
    engine = AirPadEngine(cfg, (W, H))
    for i, f in enumerate(frames):
        engine.process(f, i / 30.0)
    s = engine.stats
    infer, render = list(s.infer_ms)[10:], list(s.render_ms)[10:]
    strokes = len(engine.canvas.strokes)
    engine.close()
    print(f"frames: {len(frames)} @ {W}x{H}, strokes drawn: {strokes}")
    print(f"hand tracking : p50 {pct(infer, 50):5.1f} ms   p95 {pct(infer, 95):5.1f} ms")
    print(f"render        : p50 {pct(render, 50):5.1f} ms   p95 {pct(render, 95):5.1f} ms")
    seq_fps = 1000.0 / (np.mean(infer) + np.mean(render))
    print(f"sequential    : {seq_fps:5.1f} FPS max")

    # Pipelined: tracking on a worker thread, render on this thread (like the app).
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "bench.avi")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
        for f in frames:
            vw.write(f)
        vw.release()
        src = VideoFileSource(path)
        engine = AirPadEngine(cfg, (W, H))
        pipe = TrackingPipeline(src, engine, realtime=False)
        seq, shown = 0, 0
        t0 = time.perf_counter()
        while True:
            item = pipe.get(seq, timeout=5.0)
            if item is None:
                break
            result, seq = item
            engine.step(result)
            shown += 1
        elapsed = time.perf_counter() - t0
        pipe.stop()
        src.release()
        engine.close()
    print(f"pipelined     : {seq / elapsed:5.1f} frames tracked/s, {shown / elapsed:5.1f} frames rendered/s")
    print("(a webcam delivers 30 FPS; anything above that is headroom)")


if __name__ == "__main__":
    main()
