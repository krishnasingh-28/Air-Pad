"""Locate / download MediaPipe's hand landmark model."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_REQUIRED_MEMBERS = {"hand_detector.tflite", "hand_landmarks_detector.tflite"}


def is_valid_model(path: Path) -> bool:
    """A .task bundle is a zip holding the palm detector and landmark model."""
    try:
        with zipfile.ZipFile(path) as z:
            return _REQUIRED_MEMBERS.issubset(z.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def ensure_model(path: Path, url: str = MODEL_URL) -> Path:
    """Return ``path``, downloading the official model first if missing/corrupt."""
    path = Path(path)
    if path.is_file() and is_valid_model(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    print(f"[airpad] downloading hand model -> {path}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    if not is_valid_model(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded model from {url} is not a valid hand_landmarker bundle")
    tmp.replace(path)
    return path
