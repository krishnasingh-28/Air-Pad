"""Hand pose -> gesture classification.

Uses MediaPipe *world* landmarks (metric 3-D, origin at the hand centre), so
the decision does not depend on how far the hand is from the camera, on hand
rotation in the image plane, or on fingers pointing towards the lens.

A finger counts as *extended* when it is both straight (small total bend at
the PIP and DIP joints) and reaches away from the wrist (tip further from the
wrist than its PIP joint). Thresholds were calibrated on real MediaPipe output:

    extended fingers : curl 30-58 deg, reach 1.19-1.61
    folded fingers   : curl 73-167 deg, reach 0.60-0.99

Separate *enter* / *stay* thresholds give hysteresis so a finger hovering near
the boundary cannot flicker between states from one frame to the next.
"""

from __future__ import annotations

import enum

import numpy as np

WRIST = 0
INDEX_TIP = 8
# Palm centre = mean of wrist and the four finger MCP knuckles: a stable point
# that barely moves when fingers curl, ideal for the palm eraser.
PALM_POINTS = (0, 5, 9, 13, 17)

# (MCP, PIP, DIP, TIP) landmark indices for each finger (thumb: CMC, MCP, IP, TIP).
FINGER_JOINTS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
TRACKED_FINGERS = ("index", "middle", "ring", "pinky")  # the thumb is ignored

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class Gesture(enum.Enum):
    NONE = "none"    # no hand visible
    IDLE = "idle"    # hand visible, no command pose (fist, thumbs up, ...)
    DRAW = "draw"    # index finger only            -> pen down
    HOVER = "hover"  # index + middle (peace sign)  -> pen up, move / select
    ERASE = "erase"  # open palm (all four fingers) -> palm eraser


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = float(np.dot(u, v) / (nu * nv))
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, c)))))


def finger_metrics(world: np.ndarray, finger: str) -> tuple[float, float]:
    """Return ``(curl_deg, reach)`` for one finger.

    curl  : bend at PIP + bend at DIP (0 = perfectly straight).
    reach : |tip - wrist| / |pip - wrist| (> 1 when the finger points away).
    """
    mcp, pip, dip, tip = FINGER_JOINTS[finger]
    w = np.asarray(world, dtype=np.float64)
    curl = _angle_deg(w[pip] - w[mcp], w[dip] - w[pip]) + _angle_deg(w[dip] - w[pip], w[tip] - w[dip])
    d_pip = float(np.linalg.norm(w[pip] - w[WRIST]))
    reach = float(np.linalg.norm(w[tip] - w[WRIST])) / d_pip if d_pip > 1e-9 else 0.0
    return curl, reach


class FingerStateEstimator:
    """Per-finger extended/folded decision with hysteresis."""

    def __init__(
        self,
        enter_reach: float = 1.10,
        enter_curl: float = 65.0,
        stay_reach: float = 1.02,
        stay_curl: float = 80.0,
    ):
        if not (stay_reach <= enter_reach and stay_curl >= enter_curl):
            raise ValueError("'stay' thresholds must be looser than 'enter' thresholds")
        self.enter_reach = enter_reach
        self.enter_curl = enter_curl
        self.stay_reach = stay_reach
        self.stay_curl = stay_curl
        self._state: dict[str, bool] = {}

    def reset(self) -> None:
        self._state = {}

    def update(self, world: np.ndarray) -> dict[str, bool]:
        state = {}
        for finger in TRACKED_FINGERS:
            curl, reach = finger_metrics(world, finger)
            if self._state.get(finger, False):
                extended = reach >= self.stay_reach and curl <= self.stay_curl
            else:
                extended = reach >= self.enter_reach and curl <= self.enter_curl
            state[finger] = extended
        self._state = state
        return dict(state)


def classify(extended: dict[str, bool]) -> Gesture:
    """Map finger states to a gesture. The thumb is deliberately ignored."""
    i = extended.get("index", False)
    m = extended.get("middle", False)
    r = extended.get("ring", False)
    p = extended.get("pinky", False)
    if i and m and r and p:
        return Gesture.ERASE
    if i and m and not r and not p:
        return Gesture.HOVER
    if i and not m and not r and not p:
        return Gesture.DRAW
    return Gesture.IDLE


class GestureStabilizer:
    """Debounce: a new gesture must be seen ``confirm_frames`` frames in a row.

    While a different gesture is being confirmed, :attr:`pending` is True; the
    engine uses that to *hold* the pen (neither add points nor end the stroke),
    which removes the little tail that would otherwise be drawn while the user
    is changing pose.

    If the raw gesture keeps changing between *different* poses (e.g. HOVER,
    IDLE, HOVER while fingers are mid-motion) no single candidate would ever be
    confirmed; after ``2 * confirm_frames`` frames away from the stable gesture
    the latest raw gesture is adopted anyway, so the pen is never stuck down.
    """

    def __init__(self, confirm_frames: int = 2):
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")
        self.confirm_frames = confirm_frames
        self.stable = Gesture.NONE
        self._candidate: Gesture | None = None
        self._count = 0
        self._away = 0

    @property
    def pending(self) -> bool:
        return self._candidate is not None

    def reset(self) -> None:
        self.stable = Gesture.NONE
        self._candidate = None
        self._count = 0
        self._away = 0

    def update(self, raw: Gesture) -> Gesture:
        if raw == self.stable:
            self._candidate = None
            self._count = 0
            self._away = 0
            return self.stable
        self._away += 1
        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate = raw
            self._count = 1
        if self._count >= self.confirm_frames or self._away >= 2 * self.confirm_frames:
            self.stable = raw
            self._candidate = None
            self._count = 0
            self._away = 0
        return self.stable
