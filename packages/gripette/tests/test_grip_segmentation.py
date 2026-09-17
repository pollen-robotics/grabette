"""Tests for grip event segmentation.

Every case here maps to something measured on the real recordings or to a
failure an earlier version of this code actually produced.
"""

import math

import pytest

from gripette.grasp_projection import segment_grip

FPS = 50


def ramp(a, b, frames):
    return [a + (b - a) * i / max(frames - 1, 1) for i in range(frames)]


def hold(v, frames, wobble=0.0):
    """A held level with slow correlated drift — how a hand actually behaves.
    Deliberately NOT white noise: a sine keeps it correlated, which is what broke
    the noise-scaled thresholds this detector no longer uses."""
    return [v + wobble * math.sin(0.15 * i) for i in range(frames)]


def pick_and_lift(rest=0.18, grasp=0.52, wobble=0.01):
    """The shape of the existing datasets: approach at an intermediate posture,
    one close onto the object, then hold to the end. No release."""
    return hold(rest, 80, wobble) + ramp(rest, grasp, 10) + hold(grasp, 120, wobble)


def test_finds_exactly_one_close_in_a_pick_and_lift():
    labels, events = segment_grip(pick_and_lift())
    closes = [e for e in events if e.kind == "close"]
    assert len(closes) == 1
    assert not [e for e in events if e.kind == "open"]
    assert labels[0] == "rest" and labels[-1] == "close"


def test_the_close_is_latched_from_its_onset():
    """The command must fire when the human committed, so the robot's timing
    matches the demonstration — not when the ramp finishes."""
    labels, events = segment_grip(pick_and_lift())
    onset = events[0].onset
    assert labels[onset] == "close"
    assert labels[onset - 1] == "rest"
    assert all(v == "close" for v in labels[onset:])


def test_the_local_rest_level_is_recovered():
    """`from_c` is the LOCAL pre-ramp level, not a global constant: the closure
    histogram of real data is flat, so a dataset-wide rest value is not real."""
    _labels, events = segment_grip(pick_and_lift(rest=0.31, grasp=0.60))
    assert events[0].from_c == pytest.approx(0.31, abs=0.02)
    assert events[0].to_c == pytest.approx(0.60, abs=0.02)


def test_correlated_posture_drift_is_not_an_event():
    """The regression that killed two earlier versions. Slow drift with no hold
    accumulates past any noise-scaled bar; it must not register."""
    drift = [0.20 + 0.05 * math.sin(0.02 * i) for i in range(400)]
    _labels, events = segment_grip(drift)
    assert events == []


def test_a_close_that_is_not_held_is_not_a_grasp():
    """Holding is the discriminator that needs no noise model: a grasp is held
    while the object is transported, a twitch is not.

    Only the CLOSE side is asserted — the return leg of the twitch is a genuine
    opening motion that then holds, so reporting it as an open is correct.
    """
    twitch = hold(0.2, 60) + ramp(0.2, 0.45, 8) + ramp(0.45, 0.2, 8) + hold(0.2, 60)
    _labels, events = segment_grip(twitch)
    assert [e for e in events if e.kind == "close"] == []


def test_a_hesitation_mid_close_still_reads_as_one_event():
    """A deliberate close often pauses briefly; splitting there would report two
    half-amplitude events and neither would qualify."""
    traj = (hold(0.18, 60) + ramp(0.18, 0.34, 6) + hold(0.34, 5)
            + ramp(0.34, 0.52, 6) + hold(0.52, 120))
    _labels, events = segment_grip(traj)
    closes = [e for e in events if e.kind == "close"]
    assert len(closes) == 1
    assert closes[0].to_c == pytest.approx(0.52, abs=0.02)


def test_pick_and_place_gives_close_then_open():
    traj = (hold(0.18, 60) + ramp(0.18, 0.52, 10) + hold(0.52, 100)
            + ramp(0.52, 0.12, 10) + hold(0.12, 60))
    labels, events = segment_grip(traj)
    kinds = [e.kind for e in events]
    assert kinds == ["close", "open"]
    assert labels[0] == "rest"
    assert "close" in labels
    assert labels[-1] == "rest", "after a release the hand returns to rest"


# ---- rest-closed convention -------------------------------------------------


def test_rest_closed_labels_everything_close_except_opens():
    """With rest = full closure, rest and grasp are the SAME command, so only
    the open needs detecting and everything else is 'close'."""
    traj = (hold(0.98, 60)                      # idle, closed
            + ramp(0.98, 0.30, 10)              # open to clear the object
            + hold(0.30, 20)                    # positioning
            + ramp(0.30, 0.55, 8)               # close onto it
            + hold(0.55, 100))                  # transport
    labels, events = segment_grip(traj, rest_is_closed=True)
    assert all(e.kind == "open" for e in events)
    assert len(events) == 1
    assert labels[0] == "close", "idle-closed is the same command as a grasp"
    assert labels[-1] == "close"
    assert "open" in labels


def test_rest_closed_open_persists_while_positioning():
    """The gripper must STAY open while positioning around the object, not snap
    back to closed the moment the opening motion stops — that would collide with
    a large object."""
    traj = (hold(0.98, 40) + ramp(0.98, 0.25, 10) + hold(0.25, 40)
            + ramp(0.25, 0.50, 8) + hold(0.50, 80))
    labels, _events = segment_grip(traj, rest_is_closed=True)
    mid_positioning = 40 + 10 + 20
    assert labels[mid_positioning] == "open"


def test_rest_closed_is_immune_to_the_ambiguous_case():
    """Release-then-relax and open-then-grasp-a-wide-object are kinematically
    identical. Under this convention they must produce the same commands, so
    confusing them is harmless — that is the whole point of the convention."""
    release_then_rest = (hold(0.55, 60) + ramp(0.55, 0.15, 10) + hold(0.15, 30)
                         + ramp(0.15, 0.98, 12) + hold(0.98, 60))
    open_then_wide_grasp = (hold(0.55, 60) + ramp(0.55, 0.15, 10) + hold(0.15, 30)
                            + ramp(0.15, 0.30, 12) + hold(0.30, 60))
    a, _ = segment_grip(release_then_rest, rest_is_closed=True)
    b, _ = segment_grip(open_then_wide_grasp, rest_is_closed=True)
    # both: closed, then open, then closed again — same command sequence
    assert a[0] == b[0] == "close"
    assert a[-1] == b[-1] == "close"
    assert "open" in a and "open" in b


def test_empty_and_tiny_inputs_are_safe():
    assert segment_grip([]) == ([], [])
    labels, events = segment_grip([0.3, 0.3])
    assert labels == ["rest", "rest"] and events == []


def test_labels_align_with_the_input_length():
    for traj in (pick_and_lift(), hold(0.4, 37), ramp(0.1, 0.9, 200)):
        labels, _ = segment_grip(traj)
        assert len(labels) == len(traj)
        labels, _ = segment_grip(traj, rest_is_closed=True)
        assert len(labels) == len(traj)
