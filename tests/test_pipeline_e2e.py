"""End to end with the real hand model: photo of a pointing hand moved across
synthetic video frames -> MediaPipe -> gestures -> strokes -> rendered frames,
plus the full app loop (headless) reading a real video file."""

import cv2
import numpy as np
import pytest
from conftest import FIXTURES, MODEL

from airpad.app import main, run
from airpad.camera import CameraError, VideoFileSource, _backend_order
from airpad.config import Config, parse_args
from airpad.engine import AirPadEngine
from airpad.gestures import Gesture

W, H = 1280, 720
HAND = 560


def _hand(name):
    return cv2.resize(cv2.imread(str(FIXTURES / f"{name}.jpg")), (HAND, HAND))


def compose(hand, x, y):
    frame = np.full((H, W, 3), (120, 130, 140), np.uint8)
    frame[y:y + HAND, x:x + HAND] = hand
    return frame


def scenario():
    """(frame, label) list: write, lift (peace sign), write again, lift."""
    point, peace = _hand("pointing_up"), _hand("victory")
    frames = [(compose(point, int(x), 120), "draw") for x in np.linspace(40, 640, 46)]
    frames += [(compose(peace, 640, 120), "lift")] * 12
    frames += [(compose(point, int(x), 150), "draw") for x in np.linspace(640, 240, 31)]
    frames += [(compose(peace, 240, 150), "lift")] * 12
    return frames


@pytest.fixture(scope="module")
def run_scenario():
    cfg = Config()
    cfg.mirror = False
    engine = AirPadEngine(cfg, (W, H))
    states = []
    out = None
    for i, (frame, label) in enumerate(scenario()):
        out = engine.process(frame, i / 30.0)
        states.append((label, engine.gesture, engine.drawing))
    yield engine, states, out
    engine.close()


def test_gestures_follow_the_hand(run_scenario):
    engine, states, _ = run_scenario
    draw_ok = sum(g is Gesture.DRAW for lab, g, _ in states if lab == "draw")
    lift_ok = sum(g is Gesture.HOVER for lab, g, _ in states if lab == "lift")
    n_draw = sum(lab == "draw" for lab, _, _ in states)
    n_lift = sum(lab == "lift" for lab, _, _ in states)
    # Each acquisition of a new pose may cost: 1 coarse (fresh) detection frame that is
    # deliberately ignored + the gesture confirmation frames. Two pen-downs, two lifts.
    per_change = engine.cfg.confirm_frames + 1
    assert draw_ok >= n_draw - 2 * per_change
    assert lift_ok >= n_lift - 2 * per_change
    assert not engine.drawing


def test_two_clean_strokes_along_the_true_path(run_scenario):
    engine, _, _ = run_scenario
    strokes = engine.canvas.strokes
    assert len(strokes) == 2
    a, b = (np.array(s.points) for s in strokes)
    # Stroke 1 travels right ~600 px at a constant height; stroke 2 travels left ~400 px.
    assert a[-1, 0] - a[0, 0] > 0.8 * 600 and np.ptp(a[:, 1]) < 20
    assert b[0, 0] - b[-1, 0] > 0.8 * 400 and np.ptp(b[:, 1]) < 20
    assert abs(np.median(b[:, 1]) - np.median(a[:, 1]) - 30) < 8  # second row is 30 px lower
    tip_offset = 0.71 * HAND  # fingertip x inside the pasted photo
    assert abs(a[0, 0] - (40 + tip_offset)) < 40 and abs(a[-1, 0] - (640 + tip_offset)) < 30


def test_rendered_output_shows_ink(run_scenario):
    engine, _, out = run_scenario
    a = np.array(engine.canvas.strokes[0].points)
    mid = a[len(a) // 2]
    assert out.shape == (H, W, 3)
    assert engine.canvas.coverage()[int(mid[1]), int(mid[0])] == 255
    assert tuple(out[int(mid[1]), int(mid[0])]) == engine.color


def test_performance_budget(run_scenario):
    engine, _, _ = run_scenario
    s = engine.stats.summary()
    # 30 FPS camera = 33 ms per frame for everything. Generous bounds to stay stable on slower CPUs.
    assert s["hand_ms_p50"] < 30, s
    assert s["render_ms_p50"] < 8, s


@pytest.fixture(scope="module")
def video_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "scenario.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    assert vw.isOpened()
    frames = scenario()
    for frame, _ in frames:
        vw.write(frame)
    vw.release()
    return path, len(frames)


def test_video_file_source_reads_every_frame(video_file):
    path, n = video_file
    src = VideoFileSource(str(path))
    assert src.frame_size == (W, H)
    count, last_t = 0, -1.0
    while (item := src.read()) is not None:
        frame, t, seq = item
        assert frame.shape == (H, W, 3) and t > last_t and seq == count + 1
        last_t, count = t, count + 1
    src.release()
    assert count == n


def test_full_app_loop_headless_on_video(video_file, tmp_path):
    path, n = video_file
    cfg = parse_args(["--source", str(path), "--headless", "--no-mirror", "--save-dir", str(tmp_path)])
    summary = run(cfg)
    assert summary["frames"] == n
    assert summary["hand_ms_p50"] is not None


def test_main_reports_missing_video_cleanly(tmp_path, capsys):
    assert main(["--source", str(tmp_path / "missing.mp4"), "--headless"]) == 2
    assert "could not open" in capsys.readouterr().err


def test_cli_validation():
    with pytest.raises(SystemExit):
        parse_args(["--dim", "2"])
    with pytest.raises(SystemExit):
        parse_args(["--width", "0"])
    c = parse_args(["--camera", "1", "--board", "--infer-width", "0"])
    assert c.camera == 1 and c.start_in_board and c.infer_width == 0
    assert c.model_path == MODEL


def test_backend_order_and_missing_camera():
    order = [name for name, _ in _backend_order("auto")]
    assert order[0] in ("msmf", "any")
    assert _backend_order("dshow")[0][0] == "dshow"
    with pytest.raises(CameraError):
        VideoFileSource("definitely_not_here.mp4")
