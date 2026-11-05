import math

import numpy as np
import pytest
from synth import ScriptedTracker, observation, palm_center_px

from airpad.config import PALETTE, PEN_SIZES, Config
from airpad.engine import AirPadEngine
from airpad.gestures import Gesture

W, H = 1280, 720
DT = 1 / 30


def frames_for(seconds, cfg):
    """Frames needed to hold a pose for ``seconds`` *after* it is confirmed."""
    return math.ceil(seconds / DT) + cfg.confirm_frames + 1


class Rig:
    """Feeds scripted hands to a real engine, one frame at a time."""

    def __init__(self, tmp_path=None, size=(W, H), **cfg_overrides):
        self.cfg = Config()
        self.cfg.mirror = False
        if tmp_path is not None:
            self.cfg.save_dir = tmp_path
        for k, v in cfg_overrides.items():
            setattr(self.cfg, k, v)
        self.tracker = ScriptedTracker()
        self.engine = AirPadEngine(self.cfg, size, tracker=self.tracker)
        self.frame = np.full((size[1], size[0], 3), 90, np.uint8)
        self.t = 0.0
        self.out = None

    def step(self, obs):
        self.tracker.push(obs)
        self.out = self.engine.process(self.frame, self.t)
        self.t += DT
        return self.out

    def hold(self, pose, pt, frames):
        for _ in range(frames):
            self.step(observation(pose, pt) if pose else None)

    def move(self, pose, a, b, frames):
        for p in np.linspace(a, b, frames):
            self.step(observation(pose, p))

    @property
    def e(self):
        return self.engine

    def alpha_at(self, x, y):
        return int(self.e.canvas.coverage()[int(y), int(x)])


def test_draw_stroke_then_lift_commits_it():
    r = Rig()
    r.move("draw", (300, 400), (900, 400), 40)
    assert r.e.gesture is Gesture.DRAW and r.e.drawing
    r.hold("hover", (900, 400), 3)
    assert r.e.gesture is Gesture.HOVER and not r.e.drawing
    assert len(r.e.canvas.strokes) == 1
    s = np.array(r.e.canvas.strokes[0].points)
    assert s[:, 0].min() < 340 and s[:, 0].max() > 860
    assert np.abs(s[:, 1] - 400).max() < 3
    assert r.alpha_at(600, 400) == 255 and r.alpha_at(600, 470) == 0
    assert tuple(r.out[400, 600]) == PALETTE[0][1]


def test_first_frame_of_a_pose_does_not_draw():
    r = Rig()
    r.step(observation("draw", (500, 500)))
    assert r.e.gesture is Gesture.NONE and not r.e.drawing
    r.step(observation("draw", (500, 500)))
    assert r.e.gesture is Gesture.DRAW and r.e.drawing


def test_no_tail_while_pose_is_changing():
    r = Rig()
    r.move("draw", (300, 400), (600, 400), 20)
    n = len(r.e.canvas.active.points)
    r.step(observation("hover", (650, 380)))  # first frame of the lift: pen held
    assert r.e.drawing and len(r.e.canvas.active.points) == n
    r.step(observation("hover", (700, 360)))  # confirmed: stroke ends without the tail
    assert not r.e.drawing
    assert np.array(r.e.canvas.strokes[0].points)[:, 0].max() <= 601


def test_single_frame_misclassification_does_not_break_stroke():
    r = Rig()
    r.move("draw", (300, 400), (500, 400), 15)
    r.step(observation("three", (510, 400)))
    r.move("draw", (520, 400), (700, 400), 10)
    r.hold("hover", (700, 400), 2)
    assert len(r.e.canvas.strokes) == 1


def test_brief_tracking_dropout_keeps_the_stroke():
    r = Rig()
    r.move("draw", (300, 400), (500, 400), 15)
    r.hold(None, None, r.cfg.lost_grace_frames)
    assert r.e.drawing
    r.move("draw", (520, 400), (700, 400), 10)
    r.hold("hover", (700, 400), 2)
    assert len(r.e.canvas.strokes) == 1


def test_long_dropout_lifts_the_pen_and_resets():
    r = Rig()
    r.move("draw", (300, 400), (500, 400), 15)
    r.hold(None, None, r.cfg.lost_grace_frames + 1)
    assert not r.e.drawing and r.e.gesture is Gesture.NONE and r.e.tip is None
    assert len(r.e.canvas.strokes) == 1


def test_tracking_jump_starts_a_new_stroke_instead_of_a_long_line():
    r = Rig()
    r.move("draw", (300, 400), (400, 400), 10)
    r.hold("draw", (1100, 650), 1)  # a 700 px teleport in one frame
    r.move("draw", (1100, 650), (1150, 650), 5)
    r.hold("hover", (1150, 650), 2)
    assert len(r.e.canvas.strokes) == 2
    assert r.alpha_at(750, 525) == 0


def test_tap_makes_a_dot():
    r = Rig()
    r.hold("draw", (640, 400), 4)
    r.hold("hover", (640, 400), 2)
    assert len(r.e.canvas.strokes) == 1 and len(r.e.canvas.strokes[0].points) == 1
    assert r.alpha_at(640, 400) == 255


def test_toolbar_dwell_selects_colour_and_never_draws_there():
    r = Rig()
    b = r.e.toolbar.button("color:3")
    r.hold("draw", b.center, frames_for(r.cfg.dwell_seconds, r.cfg))
    assert r.e.color == PALETTE[3][1] and r.e.color_id == "color:3"
    assert r.e.canvas.is_empty
    r.move("draw", (400, 300), (700, 300), 15)
    r.hold("hover", (700, 300), 2)
    assert r.e.canvas.strokes[-1].color == PALETTE[3][1]


def test_toolbar_hover_shorter_than_dwell_does_nothing():
    r = Rig()
    r.hold("hover", r.e.toolbar.button("clear").center, frames_for(r.cfg.dwell_seconds, r.cfg) - 4)
    assert r.e.hover_button == "clear" and 0 < r.e.dwell.progress < 1


def test_stroke_ends_when_entering_toolbar():
    r = Rig()
    r.move("draw", (600, 400), (600, 100), 15)
    r.step(observation("draw", (600, 30)))
    assert not r.e.drawing and len(r.e.canvas.strokes) == 1


def test_open_palm_erases_under_the_palm():
    r = Rig()
    r.move("draw", (300, 400), (900, 400), 30)
    r.hold("hover", (900, 400), 2)
    assert r.alpha_at(600, 400) == 255
    obs = observation("palm", (0, 0))
    offset = np.array([600, 400]) - palm_center_px(obs)
    for x in np.linspace(-60, 60, 12):
        r.step(observation("palm", offset + [x, 0]))
    assert r.e.gesture is Gesture.ERASE
    r.hold("fist", (600, 400), 3)
    assert r.alpha_at(600, 400) == 0 and r.alpha_at(350, 400) == 255 and r.alpha_at(850, 400) == 255


def test_eraser_tool_with_index_finger():
    r = Rig()
    r.move("draw", (300, 400), (900, 400), 30)
    r.hold("hover", (900, 400), 2)
    assert r.e.handle_key(ord("e")) is None and r.e.tool == "eraser"
    r.move("draw", (550, 400), (650, 400), 10)
    r.hold("hover", (650, 400), 2)
    assert r.alpha_at(600, 400) == 0 and r.alpha_at(320, 400) == 255
    assert r.e.canvas.strokes[-1].erase


def test_custom_colour_picker_end_to_end():
    """Open picker from the toolbar, hold still on a colour, then draw with it."""
    r = Rig()
    r.hold("hover", r.e.toolbar.button("picker").center, frames_for(r.cfg.dwell_seconds, r.cfg))
    assert r.e.picker.is_open
    sx, sy, sw, sh = r.e.picker.spec_rect
    target = (sx + sw * 0.62, sy + sh * 0.5)
    want = r.e.picker.target_at(*target)[1]
    r.hold("draw", target, frames_for(r.cfg.picker_dwell_seconds, r.cfg))
    assert not r.e.picker.is_open
    assert r.e.custom_color is not None and r.e.color == r.e.custom_color and r.e.color_id == "custom"
    assert np.abs(np.array(r.e.custom_color) - want).max() <= 8
    # Still in DRAW pose where the panel was: must NOT start drawing.
    r.hold("draw", target, 5)
    assert r.e.canvas.is_empty
    # Release once (peace sign), then draw: ink is the picked colour.
    r.hold("hover", (400, 300), 3)
    r.move("draw", (400, 300), (800, 300), 20)
    r.hold("hover", (800, 300), 2)
    assert r.e.canvas.strokes[-1].color == r.e.custom_color
    assert tuple(r.out[300, 600]) == r.e.custom_color
    # The custom swatch reselects it after switching to a preset.
    r.e.do_action("color:2")
    r.e.do_action("custom")
    assert r.e.color == r.e.custom_color


def test_custom_swatch_opens_picker_when_empty_and_cancel_works():
    r = Rig()
    r.e.do_action("custom")
    assert r.e.picker.is_open
    cx, cy, cw, ch = r.e.picker.cancel_rect
    r.hold("hover", (cx + cw / 2, cy + ch / 2), frames_for(r.cfg.picker_dwell_seconds * 0.7, r.cfg))
    assert not r.e.picker.is_open and r.e.custom_color is None
    r.e.do_action("picker")
    assert r.e.handle_key(27) is None and not r.e.picker.is_open  # Esc closes, doesn't quit


def test_picker_blocks_drawing():
    r = Rig()
    r.e.do_action("picker")
    r.move("draw", (100, 650), (300, 650), 15)  # outside the panel
    assert r.e.canvas.is_empty and r.e.picker.is_open


def test_mouse_click_on_picker_and_toolbar():
    r = Rig()
    import cv2

    r.e.handle_mouse(cv2.EVENT_LBUTTONDOWN, *r.e.toolbar.button("picker").center)
    assert r.e.picker.is_open
    gx, gy, gw, gh = r.e.picker.gray_rect
    r.e.handle_mouse(cv2.EVENT_LBUTTONDOWN, gx + gw // 2, gy + 2)
    assert not r.e.picker.is_open and min(r.e.color) > 240
    r.e.handle_mouse(cv2.EVENT_LBUTTONDOWN, *r.e.toolbar.button("board").center)
    assert r.e.board
    r.e.handle_mouse(cv2.EVENT_MOUSEMOVE, *r.e.toolbar.button("board").center)
    assert r.e.board


def test_keys():
    r = Rig()
    e = r.e
    assert e.handle_key(-1) is None
    e.handle_key(ord("5"))
    assert e.color == PALETTE[4][1]
    idx = e.pen_idx
    e.handle_key(ord("]"))
    assert e.pen_idx == idx + 1
    e.handle_key(ord("["))
    e.handle_key(ord("-"))
    assert e.pen_idx == idx - 1
    for _ in range(30):
        e.handle_key(ord("="))
    assert e.pen_idx == len(PEN_SIZES) - 1
    e.handle_key(ord("B"))
    assert e.board
    e.handle_key(ord("h"))
    assert e.show_help
    assert e.handle_key(27) is None and not e.show_help  # Esc closes help first
    assert e.handle_key(ord("f")) == "fullscreen"
    assert e.handle_key(ord("q")) == "quit"
    assert e.handle_key(27) == "quit"
    e.handle_key(ord("l"))
    assert not e.show_landmarks
    e.handle_key(ord("m"))
    assert e.mirror is True  # toggled on (the rig starts with mirroring off)


def test_undo_redo_clear_via_actions():
    r = Rig()
    r.move("draw", (300, 400), (900, 400), 20)
    r.hold("hover", (900, 400), 2)
    r.e.do_action("clear")
    assert r.e.canvas.is_empty
    r.e.do_action("undo")
    assert len(r.e.canvas.strokes) == 1
    r.e.do_action("undo")
    r.e.do_action("redo")
    assert len(r.e.canvas.strokes) == 1


def test_save_action_writes_pngs(tmp_path):
    r = Rig(tmp_path)
    assert r.e.save() is None  # nothing drawn yet
    r.move("draw", (300, 400), (900, 400), 20)
    r.hold("hover", (900, 400), 2)
    paths = r.e.save()
    assert paths is not None and all(p.exists() for p in paths)
    assert {p.parent for p in paths} == {tmp_path.resolve()}


def test_board_mode_render_and_pip():
    r = Rig()
    r.e.do_action("board")
    out = r.step(observation("hover", (640, 400)))
    assert out.shape == (H, W, 3)
    assert tuple(out[400, 100]) == r.cfg.board_color
    assert tuple(out[H - 40, W - 100]) == (90, 90, 90)  # camera picture-in-picture


def test_low_resolution_camera_is_upscaled_for_crisp_ui():
    r = Rig(size=(640, 480))
    out = r.step(None)
    assert out.shape == (960, 1280, 3) and r.e.size == (1280, 960)


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 720)])
def test_ui_scales_with_display(size):
    r = Rig(size=size)
    assert r.e.toolbar.height == round(72 * size[0] / 1280)
    assert r.e.pen_size == round(PEN_SIZES[r.e.pen_idx] * size[0] / 1280)
    assert r.step(None).shape == (size[1], size[0], 3)


def test_every_cursor_and_hud_state_renders():
    r = Rig()
    for pose in ("draw", "hover", "palm", "fist"):
        r.hold(pose, (640, 400), 3)
    r.e.do_action("tool:eraser")
    r.hold("draw", (640, 400), 3)
    r.hold("hover", r.e.toolbar.button("save").center, 3)
    r.e.do_action("help")
    r.e.do_action("picker")
    r.hold("hover", (640, 400), 2)
    assert r.out.shape == (H, W, 3)


def test_rejects_bad_frames():
    r = Rig()
    with pytest.raises(ValueError):
        r.e.process(np.zeros((H, W), np.uint8), 0.0)


def test_fresh_detections_never_ink_or_change_gesture():
    r = Rig()
    r.move("draw", (300, 400), (500, 400), 12)
    n = len(r.e.canvas.active.points)
    r.step(observation("hover", (900, 200), fresh=True))  # coarse re-detection: wrong pose + far away
    assert r.e.gesture is Gesture.DRAW and len(r.e.canvas.active.points) == n
    r.move("draw", (510, 400), (600, 400), 5)
    r.hold("hover", (600, 400), 2)
    assert len(r.e.canvas.strokes) == 1
    pts = np.array(r.e.canvas.strokes[0].points)
    assert pts[:, 0].max() < 610 and np.abs(pts[:, 1] - 400).max() < 3  # no detour to (900, 200)
