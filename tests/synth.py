"""Synthetic hands and a scripted tracker for deterministic tests."""

from __future__ import annotations

import math

import numpy as np

from airpad.tracker import HandObservation

# Wrist at the origin, fingers pointing to -y (up in the image), metres.
_MCP = {
    "index": (-0.025, -0.090),
    "middle": (0.000, -0.095),
    "ring": (0.022, -0.090),
    "pinky": (0.042, -0.080),
}
_SEG = {
    "index": (0.045, 0.025, 0.022),
    "middle": (0.050, 0.030, 0.024),
    "ring": (0.046, 0.028, 0.023),
    "pinky": (0.035, 0.020, 0.020),
}
_ORDER = ("index", "middle", "ring", "pinky")
_FOLD_ANGLES = (80.0, 100.0, 70.0)  # MCP, PIP, DIP flexion of a fully folded finger


def _direction(theta_deg: float) -> np.ndarray:
    t = math.radians(theta_deg)
    return np.array([0.0, -math.cos(t), math.sin(t)])


def hand_world(extended=("index",), bend: dict | None = None) -> np.ndarray:
    """21 world landmarks. ``bend`` maps finger -> 0 (straight) .. 1 (folded)."""
    bend = dict(bend or {})
    pts = np.zeros((21, 3))
    pts[1:5] = [(-0.03, -0.03, 0.0), (-0.05, -0.05, 0.0), (-0.065, -0.07, 0.0), (-0.075, -0.09, 0.0)]
    for k, finger in enumerate(_ORDER):
        f = bend.get(finger, 0.0 if finger in extended else 1.0)
        mcp = np.array([*_MCP[finger], 0.0])
        idx = 5 + 4 * k
        pts[idx] = mcp
        theta, p = 0.0, mcp
        for j, (length, ang) in enumerate(zip(_SEG[finger], _FOLD_ANGLES, strict=True)):
            theta += f * ang
            p = p + length * _direction(theta)
            pts[idx + 1 + j] = p
    return pts


def rotate(world: np.ndarray, roll_deg: float = 0.0, pitch_deg: float = 0.0, yaw_deg: float = 0.0):
    r, p, y = (math.radians(v) for v in (roll_deg, pitch_deg, yaw_deg))
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    return world @ (rz @ rx @ ry).T


POSES = {
    "draw": ("index",),
    "hover": ("index", "middle"),
    "palm": ("index", "middle", "ring", "pinky"),
    "fist": (),
    "three": ("index", "middle", "ring"),
    "pinky": ("pinky",),
}


def observation(pose: str, tip_px, size=(1280, 720), px_per_m: float = 1100.0,
                fresh: bool = False) -> HandObservation:
    """A hand in ``pose`` whose index fingertip lands exactly on ``tip_px``."""
    world = hand_world(POSES[pose])
    xy = world[:, :2] * px_per_m
    xy = xy - xy[8] + np.asarray(tip_px, dtype=np.float64)
    norm = np.zeros((21, 3))
    norm[:, 0] = xy[:, 0] / size[0]
    norm[:, 1] = xy[:, 1] / size[1]
    return HandObservation(norm=norm, world=world, handedness="Right", score=0.99, fresh=fresh)


def palm_center_px(obs: HandObservation, size=(1280, 720)) -> np.ndarray:
    return (obs.norm[[0, 5, 9, 13, 17], :2] * np.array(size)).mean(axis=0)


class ScriptedTracker:
    """Returns pre-scripted observations, one per process() call."""

    def __init__(self, script=()):
        self.script = list(script)
        self.calls = 0

    def push(self, *obs) -> None:
        self.script.extend(obs)

    def process(self, frame, timestamp_ms):
        self.calls += 1
        return self.script.pop(0) if self.script else None

    def close(self) -> None:
        pass
