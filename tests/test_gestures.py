import pytest
from synth import POSES, hand_world, rotate

from airpad.gestures import (
    FingerStateEstimator,
    Gesture,
    GestureStabilizer,
    classify,
    finger_metrics,
)

EXPECTED = {
    "draw": Gesture.DRAW,
    "hover": Gesture.HOVER,
    "palm": Gesture.ERASE,
    "fist": Gesture.IDLE,
    "three": Gesture.IDLE,
    "pinky": Gesture.IDLE,
}


def gesture_of(world):
    return classify(FingerStateEstimator().update(world))


@pytest.mark.parametrize("pose", list(POSES))
def test_synthetic_poses(pose):
    assert gesture_of(hand_world(POSES[pose])) is EXPECTED[pose]


@pytest.mark.parametrize("pose", ["draw", "hover", "palm", "fist"])
@pytest.mark.parametrize("rot", [(0, 0, 0), (45, 0, 0), (90, 0, 0), (180, 0, 0), (-60, 0, 0),
                                 (0, 40, 0), (0, 0, 50), (30, -35, 25)])
def test_rotation_invariant(pose, rot):
    assert gesture_of(rotate(hand_world(POSES[pose]), *rot)) is EXPECTED[pose]


def test_metrics_straight_and_folded():
    w = hand_world(("index",))
    curl, reach = finger_metrics(w, "index")
    assert curl < 1 and reach > 1.2
    curl, reach = finger_metrics(w, "middle")
    assert curl > 150 and reach < 0.9


def test_thumb_is_ignored():
    w = hand_world(("index",))
    w[1:5] = w[0]  # collapse the thumb completely
    assert gesture_of(w) is Gesture.DRAW


def test_hysteresis_keeps_previous_state_in_the_ambiguous_band():
    est = FingerStateEstimator()
    half = hand_world(("index",), bend={"index": 0.42})
    curl, reach = finger_metrics(half, "index")
    assert est.stay_curl >= curl > est.enter_curl or est.stay_reach <= reach < est.enter_reach
    # Coming from extended: stays extended.
    est.update(hand_world(("index",)))
    assert est.update(half)["index"] is True
    # Coming from folded: stays folded.
    est.reset()
    est.update(hand_world(()))
    assert est.update(half)["index"] is False


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        FingerStateEstimator(enter_reach=1.0, stay_reach=1.2)


def test_stabilizer_needs_consecutive_frames():
    s = GestureStabilizer(confirm_frames=2)
    assert s.update(Gesture.DRAW) is Gesture.NONE and s.pending
    assert s.update(Gesture.DRAW) is Gesture.DRAW and not s.pending
    assert s.update(Gesture.HOVER) is Gesture.DRAW and s.pending
    assert s.update(Gesture.DRAW) is Gesture.DRAW and not s.pending  # blip rejected
    assert s.update(Gesture.HOVER) is Gesture.DRAW
    assert s.update(Gesture.HOVER) is Gesture.HOVER


def test_stabilizer_never_sticks_when_raw_keeps_changing():
    s = GestureStabilizer(confirm_frames=2)
    s.update(Gesture.DRAW)
    s.update(Gesture.DRAW)
    seq = [Gesture.HOVER, Gesture.IDLE, Gesture.HOVER, Gesture.IDLE]
    out = [s.update(g) for g in seq]
    assert out[-1] is not Gesture.DRAW


def test_stabilizer_confirm_one_is_immediate_and_reset():
    s = GestureStabilizer(confirm_frames=1)
    assert s.update(Gesture.HOVER) is Gesture.HOVER
    s.reset()
    assert s.stable is Gesture.NONE and not s.pending
    with pytest.raises(ValueError):
        GestureStabilizer(0)
