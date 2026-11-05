"""Low-latency smoothing for the tracked fingertip."""

from __future__ import annotations

import math

import numpy as np


def _smoothing_factor(cutoff_hz: float, dt: float) -> float:
    r = 2.0 * math.pi * cutoff_hz * dt
    return r / (r + 1.0)


class OneEuroFilter:
    """One Euro filter (Casiez, Roussel & Vogel, CHI 2012) for N-D points.

    An adaptive low-pass filter: when the finger is slow it smooths heavily
    (removes landmark jitter so hovering and slow writing look steady); when the
    finger moves fast the cutoff rises so the pen does not lag behind.

    The speed term uses the vector norm of the derivative, so all axes share one
    cutoff and curves are not distorted differently along x and y.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be > 0")
        if beta < 0:
            raise ValueError("beta must be >= 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    @property
    def initialized(self) -> bool:
        return self._x is not None

    def __call__(self, x, t: float) -> np.ndarray:
        """Filter sample ``x`` taken at time ``t`` (seconds). Returns a new array."""
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = float(t)
            return x.copy()
        if x.shape != self._x.shape:
            raise ValueError(f"shape changed from {self._x.shape} to {x.shape}")
        dt = float(t) - self._t
        if dt <= 0.0:
            # Duplicate or out-of-order timestamp: keep the current estimate.
            return self._x.copy()
        dx = (x - self._x) / dt
        a_d = _smoothing_factor(self.d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = _smoothing_factor(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        self._t = float(t)
        return self._x.copy()
