"""Runtime configuration and command-line parsing."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"

# 9 preset inks (BGR), selectable with keys 1-9.
PALETTE: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("white", (245, 245, 245)),
    ("yellow", (0, 215, 255)),
    ("orange", (0, 140, 255)),
    ("red", (60, 50, 235)),
    ("pink", (180, 100, 255)),
    ("purple", (230, 90, 160)),
    ("blue", (255, 140, 40)),
    ("cyan", (230, 215, 0)),
    ("green", (90, 215, 50)),
)

PEN_SIZES = (2, 3, 4, 6, 8, 11, 14, 18, 24, 32)
ERASER_SIZES = (16, 24, 32, 44, 60, 80, 110)


@dataclass
class Config:
    # Input
    camera: int = 0
    source: str | None = None          # video file instead of a webcam
    width: int = 1280
    height: int = 720
    fps: int = 30
    backend: str = "auto"              # auto | msmf | dshow | any
    mirror: bool = True
    # Display: frames narrower than this are upscaled so the UI stays crisp.
    min_display_width: int = 1280

    # Hand tracking
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL)
    infer_width: int = 640
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Gestures / pointer
    confirm_frames: int = 2            # frames a new gesture must persist
    lost_grace_frames: int = 4         # tolerate brief tracking drop-outs mid-stroke
    jump_fraction: float = 0.18        # a per-frame jump > this * width starts a new stroke
    # One Euro filter (pixel units), grid-searched on simulated handwriting with
    # 2 px landmark noise: rest jitter 2.0 -> 0.64 px, lag ~4.5 px at 700 px/s.
    filter_min_cutoff: float = 0.8
    filter_beta: float = 0.03
    filter_d_cutoff: float = 0.5
    dwell_seconds: float = 0.55        # hover time to press a toolbar button
    picker_dwell_seconds: float = 0.9  # hold-still time to pick a custom colour

    # Ink
    pen_size_index: int = 4            # into PEN_SIZES (8 px at 1280 wide)
    eraser_size_index: int = 3         # into ERASER_SIZES
    palm_eraser_size: int = 90
    min_point_dist: float = 1.5

    # Look
    dim: float = 0.75                  # camera brightness behind the ink
    board_color: tuple[int, int, int] = (0, 0, 0)  # pure black "ultra dark" page
    show_landmarks: bool = True
    start_in_board: bool = False

    # Output / run control
    save_dir: Path = field(default_factory=lambda: Path.cwd() / "outputs")
    headless: bool = False
    max_frames: int = 0                # 0 = unlimited
    run_seconds: float = 0.0           # 0 = unlimited (smoke tests)


def parse_args(argv: list[str] | None = None) -> Config:
    c = Config()
    p = argparse.ArgumentParser(
        prog="airpad",
        description="Draw and write in the air with your index finger.",
    )
    p.add_argument("--camera", type=int, default=c.camera, help="webcam index (default 0)")
    p.add_argument("--source", type=str, default=None, help="play a video file instead of a webcam")
    p.add_argument("--width", type=int, default=c.width, help="requested camera width")
    p.add_argument("--height", type=int, default=c.height, help="requested camera height")
    p.add_argument("--fps", type=int, default=c.fps, help="requested camera FPS")
    p.add_argument("--backend", choices=["auto", "msmf", "dshow", "any"], default=c.backend)
    p.add_argument("--no-mirror", dest="mirror", action="store_false", help="do not mirror the camera")
    p.add_argument("--model", dest="model_path", type=Path, default=c.model_path)
    p.add_argument("--infer-width", type=int, default=c.infer_width,
                   help="frame width fed to the hand model (0 = full size)")
    p.add_argument("--dim", type=float, default=c.dim, help="camera brightness behind ink, 0..1")
    p.add_argument("--board", dest="start_in_board", action="store_true", help="start in board mode")
    p.add_argument("--save-dir", type=Path, default=c.save_dir)
    p.add_argument("--headless", action="store_true", help="no window (benchmarks)")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--run-seconds", type=float, default=0.0, help="auto-exit after N seconds")
    a = p.parse_args(argv)
    for k, v in vars(a).items():
        setattr(c, k, v)
    if not 0.0 <= c.dim <= 1.0:
        p.error("--dim must be between 0 and 1")
    if c.width <= 0 or c.height <= 0 or c.fps <= 0:
        p.error("--width, --height and --fps must be positive")
    if c.infer_width < 0:
        p.error("--infer-width must be >= 0")
    return c
