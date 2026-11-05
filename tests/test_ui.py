import itertools

import numpy as np
import pytest

from airpad.config import PALETTE
from airpad.ui import (
    ColorPicker,
    DwellSelector,
    Toolbar,
    ToolbarState,
    bgr_to_hex,
    blend_rounded_rect,
    draw_help,
    draw_toast,
    make_gray_strip,
    make_spectrum,
)

REQUIRED = {"custom", "picker", "tool:pen", "tool:eraser", "size:-", "size:+", "undo", "redo", "clear",
            "save", "board"} | {f"color:{i}" for i in range(len(PALETTE))}


def state(**kw):
    base = {"color": PALETTE[0][1], "selected": "color:0", "custom_color": None, "tool": "pen",
            "size": 8, "board": False, "can_undo": False, "can_redo": False}
    base.update(kw)
    return ToolbarState(**base)


@pytest.mark.parametrize("width", [960, 1280, 1920, 2560])
def test_toolbar_layout_fits_and_never_overlaps(width):
    tb = Toolbar(width, PALETTE)
    assert tb.natural_width <= width
    ids = [b.id for b in tb.buttons]
    assert len(ids) == len(set(ids)) and set(ids) >= REQUIRED
    for a, b in itertools.combinations(tb.buttons, 2):
        overlap_x = a.x < b.x + b.w and b.x < a.x + a.w
        overlap_y = a.y < b.y + b.h and b.y < a.y + a.h
        assert not (overlap_x and overlap_y), (a.id, b.id)
    for b in tb.buttons:
        assert b.y >= 0 and b.y + b.h <= tb.height


def test_toolbar_hit_test():
    tb = Toolbar(1280, PALETTE)
    for b in tb.buttons:
        assert tb.hit(*b.center).id == b.id
    assert tb.hit(5, 5) is None  # margin
    assert tb.hit(640, tb.height + 5) is None
    assert tb.contains(10, tb.height - 1) and not tb.contains(10, tb.height)


def test_toolbar_render_is_cached_per_state():
    tb = Toolbar(1280, PALETTE)
    a = tb.render(state())
    assert tb.render(state()) is a
    b = tb.render(state(selected="color:3", color=PALETTE[3][1]))
    assert b is not a and b.shape == (tb.height, 1280, 3)
    tb.render(state(custom_color=(10, 200, 30), selected="custom", tool="eraser", board=True,
                    can_undo=True, can_redo=True))


def test_toolbar_overlay_and_highlight_draw_in_place():
    tb = Toolbar(1280, PALETTE)
    out = np.zeros((720, 1280, 3), np.uint8)
    tb.overlay(out, state())
    assert out[: tb.height].any() and not out[tb.height + 5:].any()
    tb.highlight(out, "save")


def test_dwell_fires_once_per_visit():
    d = DwellSelector(0.5)
    assert d.update("save", 0.0) is None
    assert d.update("save", 0.3) is None and 0.55 < d.progress < 0.65
    assert d.update("save", 0.5) == "save"
    assert d.update("save", 2.0) is None  # still hovering: no repeat
    assert d.update(None, 2.1) is None
    assert d.update("save", 2.2) is None
    assert d.update("save", 2.71) == "save"


def test_dwell_switching_target_restarts_timer():
    d = DwellSelector(0.5)
    d.update("undo", 0.0)
    d.update("redo", 0.4)
    assert d.update("redo", 0.8) is None
    assert d.update("redo", 0.9) == "redo"


def test_dwell_repeat_buttons_auto_repeat():
    d = DwellSelector(0.5, repeat=frozenset({"size:+"}), repeat_interval=0.25)
    fires = [d.update("size:+", i * 0.05) for i in range(int(1.6 / 0.05))]
    times = [i * 0.05 for i, f in enumerate(fires) if f]
    assert times[0] == pytest.approx(0.5)
    assert len(times) == 5
    assert np.allclose(np.diff(times), 0.25, atol=0.051)


def test_spectrum_and_gray_strip_content():
    spec = make_spectrum(360, 200).astype(int)
    assert spec[0].min() > 235  # top row ~ white
    mid = spec[100]
    assert (mid.max(axis=1) > 240).all() and (mid.min(axis=1) < 15).all()  # saturated hues
    assert spec[-1].max() < 40  # bottom row ~ black
    hues = {tuple(int(v > 128) for v in px) for px in mid}
    assert {(0, 0, 1), (0, 1, 0), (1, 0, 0)} <= hues  # red, green and blue all present
    g = make_gray_strip(10, 100)
    assert g[0, 0, 0] > 250 and g[-1, 0, 0] < 5 and (np.diff(g[:, 0, 0].astype(int)) <= 0).all()


def picker():
    p = ColorPicker(1280, 720, top=72, dwell=0.9)
    p.open()
    return p


@pytest.mark.parametrize("size", [(1280, 720), (1280, 960), (1920, 1080), (2560, 1440)])
def test_picker_fits_on_screen(size):
    p = ColorPicker(*size, top=round(72 * size[0] / 1280))
    x, y, w, h = p.panel
    assert x >= 0 and y >= 72 and x + w <= size[0] and y + h <= size[1]
    for r in (p.spec_rect, p.gray_rect, p.preview_rect, p.current_rect, p.cancel_rect):
        assert x <= r[0] and r[0] + r[2] <= x + w and y <= r[1] and r[1] + r[3] <= y + h


def test_picker_scales_down_but_rejects_a_too_short_window():
    ColorPicker(320, 240, top=18)  # proportionally small: fine
    with pytest.raises(ValueError):
        ColorPicker(1280, 150, top=72)


def test_picker_hold_still_picks_exact_colour_under_finger():
    p = picker()
    sx, sy, sw, sh = p.spec_rect
    pt = (sx + sw * 0.66, sy + sh * 0.5)
    expected = p.target_at(*pt)[1]
    assert p.update(pt, 0.0) is None and p.hover_color == expected
    assert p.update((pt[0] + 3, pt[1] - 2), 0.5) is None  # small jitter keeps the timer
    ev = p.update((pt[0] + 1, pt[1]), 0.95)
    assert ev is not None and ev[0] == "pick" and not p.is_open
    assert np.abs(np.array(ev[1]) - expected).max() <= 8


def test_picker_moving_restarts_hold_timer():
    p = picker()
    sx, sy, sw, sh = p.spec_rect
    p.update((sx + 50, sy + 50), 0.0)
    p.update((sx + 200, sy + 50), 0.6)  # moved far: restart
    assert p.update((sx + 200, sy + 50), 1.2) is None
    assert p.update((sx + 200, sy + 50), 1.55)[0] == "pick"


def test_picker_cancel_and_lost_finger():
    p = picker()
    cx, cy, cw, ch = p.cancel_rect
    c = (cx + cw / 2, cy + ch / 2)
    assert p.target_at(*c) == ("cancel", None)
    p.update(c, 0.0)
    p.update(None, 0.3)  # hand lost: timer resets
    assert p.progress == 0
    p.update(c, 1.0)
    assert p.update(c, 1.0 + 0.9 * 0.7 + 0.01) == ("cancel", None)


def test_picker_gray_strip_and_click():
    p = picker()
    gx, gy, gw, gh = p.gray_rect
    kind, col = p.target_at(gx + gw / 2, gy + 1)
    assert kind == "color" and min(col) > 245
    ev = p.click(gx + gw / 2, gy + gh - 1)
    assert ev[0] == "pick" and max(ev[1]) < 10 and not p.is_open
    assert p.click(10, 10) is None  # closed: ignored
    p.open()
    assert p.click(5, 5) is None and p.is_open  # outside targets: ignored


def test_picker_renders_without_error():
    p = picker()
    out = np.zeros((720, 1280, 3), np.uint8)
    sx, sy, sw, sh = p.spec_rect
    p.update((sx + 10, sy + 10), 0.0)
    p.render(out, (0, 0, 255))
    assert out.any()


def test_overlays_render_and_clip():
    out = np.zeros((720, 1280, 3), np.uint8)
    draw_help(out, 72, 1.0)
    draw_toast(out, "Saved airpad_20260101_120000.png", 72, 1.0)
    draw_toast(out, "x" * 400, 72, 1.0)  # wider than the screen: clipped, no crash
    blend_rounded_rect(out, (-50, -50, 100, 100), 20, (255, 255, 255), 0.5)
    assert out.any()


def test_hex():
    assert bgr_to_hex((255, 128, 0)) == "#0080FF"


def test_board_mode_toolbar_matches_the_page():
    from airpad.ui import PANEL_BG

    tb = Toolbar(1280, PALETTE, board_bg=(0, 0, 0))
    out = np.full((720, 1280, 3), 200, np.uint8)
    tb.overlay(out, state(board=True))
    assert tuple(out[3, 3]) == (0, 0, 0)  # margin pixel: page colour, fully opaque
    assert out[: tb.height].max() > 200  # buttons still drawn (white labels/swatches)
    assert np.array_equal(out[: tb.height], tb.render(state(board=True)))
    cam = tb.render(state(board=False))
    assert tuple(cam[3, 3]) == PANEL_BG  # camera mode keeps its own bar
