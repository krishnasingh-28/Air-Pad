"""Stroke smoothing: centripetal Catmull-Rom interpolation.

Fingertip samples arrive at camera rate (~30 Hz), so fast handwriting gives
sparse points and a plain polyline looks angular. A Catmull-Rom spline passes
exactly through every sample (no extra lag or shape drift) and fills the gaps
with a smooth curve. The centripetal variant (alpha = 0.5) never produces
cusps or self-intersections inside a segment.
"""

from __future__ import annotations

import numpy as np


def dedupe(points: np.ndarray, min_dist: float = 1e-3) -> np.ndarray:
    """Drop points closer than ``min_dist`` to their predecessor (keeps the first).

    Vectorised because it runs on the live stroke every frame.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    keep = np.empty(len(pts), dtype=bool)
    keep[0] = True
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) >= min_dist
    return pts[keep]


def smooth_slow_segments(points, dense: float = 4.0, sparse: float = 12.0) -> np.ndarray:
    """Speed-adaptive [1, 2, 1] / 4 smoothing of a stroke's interior points.

    Slow writing produces densely spaced samples where residual landmark
    jitter shows up as wobble; there each interior point is fully smoothed
    with its neighbours. Fast writing produces sparse samples where corners
    (V, N, M ...) are real shape, so points whose neighbours are more than
    ``sparse`` px away are left untouched. Linear blend in between. End
    points never move, so the stroke stays attached to the fingertip.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return pts.copy()
    d = np.diff(pts, axis=0)
    seg = np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])  # element-wise: deterministic
    spacing = 0.5 * (seg[:-1] + seg[1:])  # mean neighbour distance per interior point
    k = 0.25 * np.clip((sparse - spacing) / (sparse - dense), 0.0, 1.0)
    out = pts.copy()
    out[1:-1] += k[:, None] * (pts[:-2] + pts[2:] - 2.0 * pts[1:-1])
    return out


def catmull_rom(
    points,
    spacing: float = 2.0,
    alpha: float = 0.5,
    max_subdivisions: int = 32,
    *,
    remove_duplicates: bool = True,
    return_offsets: bool = False,
):
    """Interpolate ``points`` (N x 2) with a centripetal Catmull-Rom spline.

    Each segment is subdivided so consecutive output samples are roughly
    ``spacing`` pixels apart. The output starts at the first point, ends at the
    last point and passes through every input point. Fewer than three points
    are returned unchanged (a straight line is already exact).

    With ``return_offsets`` also returns ``offsets`` (len = segments + 1): the
    samples of segment ``i`` are ``out[offsets[i]:offsets[i + 1]]`` and
    ``offsets[-1]`` indexes the final point. Every sample is computed with
    element-wise maths from its own four control points only, so a segment's
    samples are bit-identical whichever window of a stroke they come from.
    Callers relying on that must pass ``remove_duplicates=False`` (zero-length
    segments are still handled: the result stays finite).
    """
    if spacing <= 0:
        raise ValueError("spacing must be > 0")
    pts = dedupe(points) if remove_duplicates else np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 3:
        out = pts.copy()
        return (out, np.arange(max(n, 1))) if return_offsets else out

    # Reflect the end points so the first and last segments are curved too.
    padded = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    p0, p1, p2, p3 = padded[:-3], padded[1:-2], padded[2:-1], padded[3:]

    eps = 1e-9

    def dist(a, b):
        d = a - b
        return np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1])

    def knot(d):
        # sqrt is correctly rounded on every SIMD/scalar path; pow is not
        # guaranteed to be, and stroke replay needs bit-identical results.
        return np.sqrt(d) if alpha == 0.5 else d**alpha

    d01, d12, d23 = dist(p1, p0), dist(p2, p1), dist(p3, p2)
    t1 = np.maximum(knot(d01), eps)
    t2 = t1 + np.maximum(knot(d12), eps)
    t3 = t2 + np.maximum(knot(d23), eps)

    counts = np.clip(np.ceil(d12 / spacing), 1, max_subdivisions).astype(np.int64)
    seg = np.repeat(np.arange(n - 1), counts)
    starts = np.cumsum(counts) - counts
    u = (np.arange(int(counts.sum())) - starts[seg]) / counts[seg]

    T1, T2, T3 = t1[seg], t2[seg], t3[seg]
    q0, q1, q2, q3 = p0[seg], p1[seg], p2[seg], p3[seg]
    t = T1 + (T2 - T1) * u

    def lerp(a, b, ta, tb):
        w = ((t - ta) / (tb - ta))[:, None]
        return a + (b - a) * w

    # Barry-Goldman pyramidal evaluation (t0 = 0).
    a1 = lerp(q0, q1, 0.0, T1)
    a2 = lerp(q1, q2, T1, T2)
    a3 = lerp(q2, q3, T2, T3)
    b1 = lerp(a1, a2, 0.0, T2)
    b2 = lerp(a2, a3, T1, T3)
    c = lerp(b1, b2, T1, T2)
    out = np.vstack([c, pts[-1:]])
    if return_offsets:
        return out, np.concatenate([[0], np.cumsum(counts)])
    return out
