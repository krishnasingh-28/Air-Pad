import numpy as np
import pytest

from airpad.spline import catmull_rom, dedupe, smooth_slow_segments


def test_passes_through_every_input_point():
    rng = np.random.default_rng(3)
    pts = np.cumsum(rng.normal(0, 15, (40, 2)), axis=0) + 400
    out = catmull_rom(pts, spacing=2.0)
    for p in pts:
        assert np.min(np.linalg.norm(out - p, axis=1)) < 1e-6
    assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])


def test_dense_output_spacing():
    t = np.linspace(0, np.pi, 12)
    pts = np.c_[300 + 200 * np.cos(t), 300 + 200 * np.sin(t)]
    out = catmull_rom(pts, spacing=2.0)
    gaps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert gaps.max() <= 2 * 2.0 + 1e-9


def test_follows_a_circle_closely():
    t = np.linspace(0, 2 * np.pi, 25)  # only 25 samples: a polyline would be visibly faceted
    pts = np.c_[500 + 150 * np.cos(t), 400 + 150 * np.sin(t)]
    out = catmull_rom(pts, spacing=1.0)
    r = np.linalg.norm(out - [500, 400], axis=1)
    assert np.abs(r - 150).max() < 0.01 * 150


@pytest.mark.parametrize("n", [0, 1, 2])
def test_short_input_unchanged(n):
    pts = np.arange(2 * n, dtype=float).reshape(n, 2)
    assert np.array_equal(catmull_rom(pts), pts)


def test_duplicates_collinear_and_reversals_are_finite():
    pts = np.array([[0, 0], [0, 0], [10, 0], [10, 0], [20, 0], [10, 0], [10, 0.0000001], [30, 5]])
    out = catmull_rom(pts)
    assert np.isfinite(out).all()
    assert len(out) >= 4


def test_dedupe_drops_consecutive_duplicates():
    pts = np.array([[1, 1], [1, 1], [2, 2], [2, 2], [1, 1]], dtype=float)
    assert np.array_equal(dedupe(pts), [[1, 1], [2, 2], [1, 1]])


def test_invalid_spacing():
    with pytest.raises(ValueError):
        catmull_rom([[0, 0], [1, 1], [2, 0]], spacing=0)


def test_adaptive_smoothing_reduces_wobble_on_slow_strokes():
    rng = np.random.default_rng(5)
    x = np.arange(0, 300, 3.0)  # 3 px spacing = slow writing
    noisy = np.c_[x, 200 + rng.normal(0, 1.3, len(x))]
    out = smooth_slow_segments(noisy)
    assert out[1:-1, 1].std() < 0.75 * noisy[1:-1, 1].std()


def test_adaptive_smoothing_keeps_fast_sharp_corners():
    v = np.array([[0, 0], [20, 40], [40, 0], [60, 40]], dtype=float)  # 45 px spacing = fast
    assert np.array_equal(smooth_slow_segments(v), v)


def test_adaptive_smoothing_keeps_endpoints_and_short_input():
    rng = np.random.default_rng(6)
    pts = np.cumsum(rng.normal(0, 2, (30, 2)), axis=0)
    out = smooth_slow_segments(pts)
    assert np.array_equal(out[[0, -1]], pts[[0, -1]])
    two = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert np.array_equal(smooth_slow_segments(two), two)
