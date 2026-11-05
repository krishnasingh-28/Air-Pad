"""Real MediaPipe inference on real hand photos (no mocks)."""

import cv2
import numpy as np
import pytest
from conftest import FIXTURES, MODEL

from airpad.gestures import FingerStateEstimator, Gesture, classify
from airpad.tracker import HandTracker


def load(name):
    img = cv2.imread(str(FIXTURES / f"{name}.jpg"))
    assert img is not None, name
    return img


def detect(img, infer_width=0, frames=3):
    """Feed a still photo as a short video (VIDEO mode tracks across frames)."""
    with HandTracker(MODEL, infer_width=infer_width) as tr:
        obs = None
        for i in range(frames):
            obs = tr.process(img, i * 33)
        return obs


@pytest.mark.parametrize(
    "name, expected",
    [
        ("pointing_up", Gesture.DRAW),
        ("victory", Gesture.HOVER),
        ("thumbs_up", Gesture.IDLE),
        ("thumbs_down", Gesture.IDLE),
        ("woman_hands", Gesture.ERASE),
    ],
)
def test_real_photo_gestures(name, expected):
    obs = detect(load(name))
    assert obs is not None, f"no hand found in {name}"
    assert classify(FingerStateEstimator().update(obs.world)) is expected


def test_mirrored_photos_classify_the_same():
    for name, expected in (("pointing_up", Gesture.DRAW), ("victory", Gesture.HOVER)):
        obs = detect(cv2.flip(load(name), 1))
        assert classify(FingerStateEstimator().update(obs.world)) is expected


def test_fingertip_location_on_pointing_photo():
    obs = detect(load("pointing_up"))
    tip = obs.norm[8, :2]
    assert abs(tip[0] - 0.71) < 0.04 and abs(tip[1] - 0.07) < 0.04, tip


def test_downscaled_inference_matches_full_resolution():
    big = cv2.resize(load("pointing_up"), (1280, 1280))
    full = detect(big, infer_width=0).norm[8, :2]
    small = detect(big, infer_width=640).norm[8, :2]
    assert np.abs(full - small).max() < 0.01  # < 13 px at 1280


def test_no_hand_returns_none_and_duplicate_timestamps_are_safe():
    blank = np.full((480, 640, 3), 127, np.uint8)
    with HandTracker(MODEL) as tr:
        assert tr.process(blank, 100) is None
        assert tr.process(blank, 100) is None  # same timestamp: auto-bumped, no crash
        assert tr.process(blank, 50) is None  # going backwards: also safe


# ---------------------------------------------------------------- tile fallback
W, H, PHOTO = 1280, 720, 560


def compose(name, x, y=120):
    f = np.full((H, W, 3), (120, 130, 140), np.uint8)
    f[y:y + PHOTO, x:x + PHOTO] = cv2.resize(load(name), (PHOTO, PHOTO))
    return f


def test_first_detection_is_flagged_fresh_then_tracked():
    frame = compose("pointing_up", 300)
    with HandTracker(MODEL) as tr:
        results = [tr.process(frame, i * 33) for i in range(6)]
    assert results[0] is not None  # the full-frame detector goes first on a cold start
    assert [o.fresh for o in results] == [True, False, False, False, False, False]


def test_tile_fallback_finds_hand_the_full_frame_detector_misses():
    frame = compose("victory", 640)  # measured: full-frame palm detector misses this
    with HandTracker(MODEL, tile_search=False) as tr:
        assert all(tr.process(frame, i * 33) is None for i in range(8))
    with HandTracker(MODEL) as tr:
        results = [tr.process(frame, i * 33) for i in range(14)]
    found = [o for o in results if o is not None]
    assert len(found) >= 8 and tr.last_source == "tile"
    tracked = [o for o in found if not o.fresh]
    g = {classify(FingerStateEstimator().update(o.world)) for o in tracked}
    assert g == {Gesture.HOVER}  # the coarse first detection (wrong pose) is flagged, not used


def _run_sweep(tr, xs):
    tips, modes = [], []
    for i, x in enumerate(xs):
        o = tr.process(compose("victory", int(x)), i * 33)
        modes.append(tr.mode)
        tips.append(None if o is None or o.fresh else o.norm[8, :2] * (W, H))
    usable = [t for t in tips if t is not None]
    pairs = zip(tips[:-1], tips[1:], strict=True)
    steps = [np.hypot(*(b - a)) for a, b in pairs if a is not None and b is not None]
    return usable, steps, modes


def test_tracking_stays_continuous_when_switching_between_trackers():
    """Hard hand: found by a tile, later taken over by the primary - no jumps."""
    xs = np.linspace(640, 720, 20).tolist() + np.linspace(720, 0, 60).tolist()
    with HandTracker(MODEL) as tr:
        usable, steps, modes = _run_sweep(tr, xs)
    assert len(usable) >= 0.9 * len(xs)
    assert max(steps) < 30, max(steps)
    assert len(set(modes)) >= 2


def test_tile_to_tile_handover_is_continuous(monkeypatch):
    """Primary disabled: the hand must be handed across all tiles without gaps or jumps."""
    xs = np.linspace(720, 0, 90)  # ~8 px/frame, right edge to left edge
    with HandTracker(MODEL) as tr:
        monkeypatch.setattr(tr, "_primary_detect", lambda frame, ts: None)
        usable, steps, modes = _run_sweep(tr, xs)
    assert len(usable) >= 0.9 * len(xs)
    assert max(steps) < 30, max(steps)
    tiles_used = [m for i, m in enumerate(modes) if m.startswith("tile") and (i == 0 or modes[i - 1] != m)]
    assert len(set(tiles_used)) >= 3, modes  # handed over across every tile


def test_square_frames_use_no_tiles():
    sq = cv2.resize(load("pointing_up"), (720, 720))
    with HandTracker(MODEL) as tr:
        tr.process(sq, 0)
        assert tr._tiles == []


def test_idle_search_runs_at_most_one_detector_per_frame(monkeypatch):
    import airpad.tracker as trk

    calls = []
    real = trk._VideoLandmarker.detect

    def counting(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(trk._VideoLandmarker, "detect", counting)
    blank = np.full((H, W, 3), (120, 130, 140), np.uint8)
    with HandTracker(MODEL, frame_size=(W, H)) as tr:
        per_frame = []
        for i in range(24):
            calls.clear()
            assert tr.process(blank, i * 33) is None
            per_frame.append(len(calls))
    assert max(per_frame) == 1


def test_hard_hand_found_quickly_from_a_cold_start():
    frame = compose("victory", 640)
    with HandTracker(MODEL, frame_size=(W, H)) as tr:
        first = next(i for i in range(12) if tr.process(frame, i * 33) is not None)
    assert first <= 6  # full tile sweep is 6 frames (~0.2 s)
