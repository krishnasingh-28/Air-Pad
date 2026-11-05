import cv2
import numpy as np
import pytest

import airpad.canvas as canvas_mod
from airpad.canvas import Canvas, Stroke, render_stroke

W, H = 320, 200
RED = (40, 40, 230)
BLUE = (230, 90, 30)


def draw(canvas, pts, color=RED, size=8, erase=False, live=False):
    """Draw a stroke. ``live=True`` renders a frame after every point, like the app."""
    canvas.begin(color, size, erase)
    for x, y in pts:
        canvas.add_point(x, y)
        if live:
            canvas.layers()
    return canvas.end()


def line(x0, x1, y, n=30):
    return [(x, y) for x in np.linspace(x0, x1, n)]


def scribble(seed, n=80, start=(160, 100), step=6):
    rng = np.random.default_rng(seed)
    ang = np.cumsum(rng.normal(0, 0.5, n))
    pts = np.cumsum(np.c_[np.cos(ang), np.sin(ang)] * step, axis=0) + start
    return [tuple(p) for p in pts]


def bg(value=100):
    return np.full((H, W, 3), value, np.uint8)


def snapshot(c):
    color, inv = c.layers()
    return color.copy(), inv.copy()


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_empty_composite_is_background_dimmed():
    c = Canvas(W, H)
    b = bg(200)
    assert np.array_equal(c.composite(b), b)
    out = c.composite(b, dim=0.5)
    assert np.all(np.abs(out.astype(int) - 100) <= 1)
    assert c.composite(b) is not b  # never returns the caller's buffer


def test_pen_stroke_inks_its_path_only():
    c = Canvas(W, H)
    draw(c, line(40, 280, 100))
    out = c.composite(bg())
    assert tuple(out[100, 160]) == RED
    assert tuple(out[30, 160]) == (100, 100, 100)
    assert len(c.strokes) == 1 and c.can_undo and not c.can_redo


@pytest.mark.parametrize("live", [False, True])
def test_composite_matches_float_reference_everywhere(live):
    c = Canvas(W, H)
    draw(c, line(40, 280, 100), RED, 9, live=live)
    c.begin(BLUE, 5)
    for i in range(30):  # leave this one live (uncommitted)
        c.add_point(160, 40 + i * 4)
        c.layers()
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    color, inv = c.layers()
    dim = 0.7
    ref = color.astype(np.float64) + frame.astype(np.float64) * inv.astype(np.float64) / 255 * dim
    out = c.composite(frame, dim)
    assert np.abs(out.astype(np.float64) - np.clip(ref, 0, 255)).max() <= 1.0


def test_composite_outside_ink_box_is_just_the_dimmed_background():
    c = Canvas(W, H)
    draw(c, line(100, 140, 50), RED, 6)
    x0, y0, x1, y1 = c.ink_rect
    assert x1 - x0 < 70 and y1 - y0 < 20  # tight box
    frame = np.random.default_rng(1).integers(0, 256, (H, W, 3), dtype=np.uint8)
    out = c.composite(frame, 0.6)
    mask = np.ones((H, W), bool)
    mask[y0:y1, x0:x1] = False
    assert np.array_equal(out[mask], cv2.convertScaleAbs(frame, alpha=0.6)[mask])


def test_antialiased_edges_exist():
    c = Canvas(W, H)
    draw(c, [(20, 20), (300, 180)], RED, 6)
    vals = np.unique(c.coverage())
    assert ((vals > 0) & (vals < 255)).any()


def test_eraser_removes_ink():
    c = Canvas(W, H)
    draw(c, line(40, 280, 100), RED, 8)
    draw(c, line(100, 200, 100), (0, 0, 0), 30, erase=True, live=True)
    cov = c.coverage()
    assert cov[100, 150] == 0 and cov[100, 60] == 255


def test_erase_on_empty_canvas_is_not_recorded():
    c = Canvas(W, H)
    assert draw(c, line(10, 100, 50), erase=True, live=True) is None
    assert not c.strokes and not c.can_undo
    assert not c.alpha().any()


def test_tap_draws_a_round_dot():
    c = Canvas(W, H)
    draw(c, [(160, 100)], RED, 20)
    cov = c.coverage()
    area = (cov > 127).sum()
    assert abs(area - np.pi * 10**2) < 0.15 * np.pi * 10**2
    assert cov[100, 160] == 255 and cov[100, 175] == 0


def test_stroke_add_respects_min_distance():
    s = Stroke((0, 0, 0), 4)
    assert s.add(0, 0) and not s.add(0.5, 0.5, min_dist=1.0) and s.add(2, 0, min_dist=1.0)
    assert len(s.points) == 2


def test_live_stroke_is_visible_but_not_in_history_until_end():
    c = Canvas(W, H)
    c.begin(RED, 8)
    for p in line(40, 280, 100):
        c.add_point(*p)
        c.layers()
    assert c.coverage()[100, 160] == 255
    assert not c.strokes and not c.can_undo and not c.is_empty
    c.end()
    assert c.alpha()[100, 160] == 255 and c.can_undo


def test_cancel_removes_frozen_parts_of_live_stroke():
    c = Canvas(W, H)
    draw(c, line(40, 280, 40), BLUE)
    before = snapshot(c)
    c.begin(RED, 8)
    for p in line(40, 280, 120):
        c.add_point(*p)
        c.layers()
    assert c._frozen > 10  # most of it was frozen into the committed layers
    c.cancel()
    assert same(snapshot(c), before)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_live_view_equals_committed_equals_rerender_bit_exact(seed):
    c = Canvas(W, H)
    draw(c, scribble(seed + 10, n=40), BLUE, 5, live=True)
    c.begin(RED, 9)
    for p in scribble(seed, n=120):
        c.add_point(*p)
        last_live = snapshot(c)
    c.end()
    assert len(c.strokes[-1].chunks) > 5  # really exercised the incremental path
    committed = snapshot(c)
    assert same(committed, last_live)  # what you saw is exactly what got committed
    c.undo()
    c.redo()  # redo re-renders the stroke from its points + recorded chunks
    assert same(snapshot(c), committed)
    c._rerender()  # full re-render of everything
    assert same(snapshot(c), committed)


def test_live_cost_is_constant_per_frame(monkeypatch):
    """Per-frame drawing work must not grow with the stroke length."""
    drawn = []
    real = cv2.polylines

    def counting(img, pts, *a, **k):
        drawn.append(sum(len(p) for p in pts))
        return real(img, pts, *a, **k)

    monkeypatch.setattr(canvas_mod.cv2, "polylines", counting)
    c = Canvas(1280, 720)
    c.begin(RED, 8)
    per_frame = []
    for p in scribble(3, n=400, start=(640, 360), step=4):
        drawn.clear()
        c.add_point(*p)
        c.layers()
        per_frame.append(sum(drawn))
    early, late = np.mean(per_frame[20:60]), np.mean(per_frame[-40:])
    assert late < 1.5 * early + 5, (early, late)
    assert max(per_frame[20:]) < 80


def test_add_point_without_begin_raises():
    with pytest.raises(RuntimeError):
        Canvas(W, H).add_point(1, 1)


def test_undo_redo_clear_round_trip():
    c = Canvas(W, H)
    s0 = snapshot(c)
    draw(c, line(40, 280, 60), RED, live=True)
    s1 = snapshot(c)
    draw(c, line(40, 280, 140), BLUE, live=True)
    s2 = snapshot(c)
    assert c.clear()
    assert same(snapshot(c), s0)
    assert c.undo()  # undo clear
    assert same(snapshot(c), s2)
    assert c.undo()
    assert same(snapshot(c), s1)
    assert c.undo()
    assert same(snapshot(c), s0)
    assert not c.undo()
    assert c.redo() and c.redo()
    assert same(snapshot(c), s2)
    assert c.redo()  # redo clear
    assert not c.strokes and not c.redo()
    assert c.ink_rect is None


def test_new_stroke_after_undo_drops_redo():
    c = Canvas(W, H)
    draw(c, line(40, 280, 60))
    c.undo()
    assert c.can_redo
    draw(c, line(40, 280, 120))
    assert not c.can_redo


def test_clear_empty_is_noop():
    c = Canvas(W, H)
    assert not c.clear() and not c.can_undo


def test_many_strokes_incremental_equals_full_rerender():
    rng = np.random.default_rng(2)
    c = Canvas(W, H)
    for i in range(12):
        pts = scribble(100 + i, n=int(rng.integers(1, 60)), step=5)
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        draw(c, pts, col, int(rng.integers(2, 20)), erase=i % 5 == 4, live=bool(i % 2))
    inc = snapshot(c)
    c._rerender()
    assert same(snapshot(c), inc)


def test_degenerate_zigzag_stays_finite():
    c = Canvas(W, H)
    pts = [(100, 100), (102, 100), (100, 100), (102, 100), (100, 100), (102, 100), (101, 101)]
    draw(c, pts, RED, 6, live=True)
    assert c.alpha()[100, 101] == 255


def test_to_bgra_unpremultiplies():
    c = Canvas(W, H)
    draw(c, line(40, 280, 100), BLUE, 12)
    bgra = c.to_bgra()
    assert tuple(bgra[100, 160]) == (*BLUE, 255)
    assert bgra[10, 10, 3] == 0
    edge = (bgra[..., 3] > 0) & (bgra[..., 3] < 255)
    assert edge.any()
    # Below ~25% coverage the 8-bit premultiplied value is too coarse to recover the
    # exact colour (and those pixels are nearly transparent anyway).
    visible_edge = (bgra[..., 3] >= 64) & (bgra[..., 3] < 255)
    assert visible_edge.any()
    assert np.abs(bgra[visible_edge][:, :3].astype(int) - BLUE).max() <= 12


def test_to_board_on_background_colour():
    c = Canvas(W, H)
    draw(c, line(40, 280, 100), RED)
    board = c.to_board((30, 30, 30))
    assert tuple(board[100, 160]) == RED and tuple(board[10, 10]) == (30, 30, 30)


def test_save_writes_both_pngs_even_to_unicode_paths(tmp_path):
    c = Canvas(W, H)
    draw(c, line(40, 280, 100), RED)
    d = tmp_path / "sortie_été"
    board, transparent = c.save(d, (30, 30, 30))
    b = cv2.imdecode(np.fromfile(str(board), np.uint8), cv2.IMREAD_UNCHANGED)
    t = cv2.imdecode(np.fromfile(str(transparent), np.uint8), cv2.IMREAD_UNCHANGED)
    assert b.shape == (H, W, 3) and t.shape == (H, W, 4)
    assert tuple(t[100, 160]) == (*RED, 255)
    board2, _ = c.save(d, (30, 30, 30))
    assert board2 != board  # same-second saves never overwrite


def test_render_stroke_ignores_empty_stroke():
    color = np.zeros((10, 10, 3), np.uint8)
    inv = np.full_like(color, 255)
    assert render_stroke(color, inv, Stroke((1, 2, 3), 4)) is None
    assert (inv == 255).all()


def test_background_shape_mismatch_raises():
    with pytest.raises(ValueError):
        Canvas(W, H).composite(np.zeros((H + 1, W, 3), np.uint8))
