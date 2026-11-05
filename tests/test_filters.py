import numpy as np
import pytest

from airpad.config import Config
from airpad.filters import OneEuroFilter

FPS = 30.0


def make(**kw):
    c = Config()
    params = {"min_cutoff": c.filter_min_cutoff, "beta": c.filter_beta, "d_cutoff": c.filter_d_cutoff}
    params.update(kw)
    return OneEuroFilter(**params)


def test_first_sample_passes_through():
    f = make()
    out = f([100.0, 200.0], 0.0)
    assert np.allclose(out, [100, 200])
    assert f.initialized


def test_removes_jitter_when_finger_is_still():
    rng = np.random.default_rng(0)
    f = make()
    raw, filt = [], []
    for i in range(int(3 * FPS)):
        x = np.array([640.0, 360.0]) + rng.normal(0, 2.0, 2)
        raw.append(x)
        filt.append(f(x, i / FPS))
    raw, filt = np.array(raw[15:]), np.array(filt[15:])
    assert filt.std(axis=0).max() < 0.4 * raw.std(axis=0).min()


def test_low_lag_when_writing_fast():
    f = make()
    speed = 800.0  # px/s - brisk handwriting
    for i in range(int(FPS)):
        t = i / FPS
        out = f([100 + speed * t, 300.0], t)
    lag_px = (100 + speed * (FPS - 1) / FPS) - out[0]
    assert 0 <= lag_px < speed / FPS * 0.5, lag_px  # under half a frame of motion


def test_noisy_handwriting_shape_is_closer_to_true_path_than_raw():
    """Shape error = distance to the true *path* (what you see on screen)."""
    rng = np.random.default_rng(1)
    f = make()
    dense_t = np.linspace(0, 4, 40000)
    path = np.c_[640 + 220 * np.cos(2.6 * dense_t), 360 + 110 * np.sin(5.2 * dense_t)]
    raw, filt = [], []
    for i in range(int(4 * FPS)):
        t = i / FPS
        truth = np.array([640 + 220 * np.cos(2.6 * t), 360 + 110 * np.sin(5.2 * t)])
        meas = truth + rng.normal(0, 2.0, 2)
        out = f(meas, t)
        if i > 15:
            raw.append(meas)
            filt.append(out)

    def shape_err(pts):
        d = np.sqrt(((np.array(pts)[:, None, :] - path[None, ::8, :]) ** 2).sum(-1))
        return d.min(axis=1).mean()

    assert shape_err(filt) < 0.85 * shape_err(raw)


def test_duplicate_or_backwards_timestamp_keeps_estimate():
    f = make()
    f([0.0, 0.0], 1.0)
    a = f([10.0, 0.0], 1.1)
    b = f([500.0, 500.0], 1.1)
    c = f([500.0, 500.0], 1.05)
    assert np.allclose(a, b) and np.allclose(a, c)


def test_reset_forgets_history():
    f = make()
    f([0.0, 0.0], 0.0)
    f([5.0, 5.0], 0.1)
    f.reset()
    assert not f.initialized
    assert np.allclose(f([300.0, 40.0], 0.2), [300, 40])


def test_output_is_a_copy():
    f = make()
    out = f([1.0, 2.0], 0.0)
    out[0] = 99
    assert np.allclose(f([1.0, 2.0], 0.1), [1, 2])


@pytest.mark.parametrize("kw", [{"min_cutoff": 0}, {"d_cutoff": -1}, {"beta": -0.1}])
def test_invalid_parameters(kw):
    with pytest.raises(ValueError):
        make(**kw)


def test_shape_change_raises():
    f = make()
    f([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        f([1.0, 2.0, 3.0], 0.1)
