"""A goal exactly AT the joint limit must be accepted, not rejected.

Found on hardware: commanding a full close (93.5 deg = the configured maximum)
was refused with "Motor 1 goal 1.632 rad outside limits [0.000, 1.632]". Goals
cross gRPC as proto `float` (float32) while the limits are float64, so a value
equal to the limit quantises to marginally above it.

This matters specifically because of the grasp projection: `decode(s, c=1)`
targets the limit BY DESIGN — a full close is meant to drive into the stop. So
the boundary case is the normal case, not an edge case.
"""

import math
import struct
import sys
import types

import pytest

if "serial" not in sys.modules:
    _stub = types.ModuleType("serial")
    _stub.Serial = object
    _stub.SerialException = Exception
    sys.modules["serial"] = _stub

from gripette.hardware.motors import MotorController  # noqa: E402

LIM1 = math.radians(93.5)
LIM2 = math.radians(116.0)


def as_float32(x: float) -> float:
    """Round-trip through float32, as a proto `float` field does on the wire."""
    return struct.unpack("f", struct.pack("f", x))[0]


def make():
    """A controller with the real limits. Mock mode engages by itself when
    rustypot is absent, which is the case off the Pi — so no bus is touched."""
    m = MotorController(limits=((0.0, LIM1), (0.0, LIM2)))
    assert m._mock, "expected mock mode off-hardware; this test must not open a bus"
    return m


def test_the_float32_round_trip_really_does_overshoot():
    """Establish the premise rather than assume it: if float32(limit) were <= the
    limit, this whole tolerance would be unnecessary."""
    assert as_float32(LIM1) > LIM1, "premise: float32 rounds the limit UP here"


def test_a_goal_exactly_at_the_limit_is_accepted():
    """The regression. Before the tolerance this raised."""
    m = make()
    m.write_goal_positions(as_float32(LIM1), 0.0)
    assert m._mock_positions[0] == pytest.approx(LIM1, abs=1e-4)


def test_the_accepted_goal_is_clamped_into_range():
    """Accepting it must not pass an out-of-range value down to the bus."""
    m = make()
    m.write_goal_positions(as_float32(LIM1), as_float32(LIM2))
    assert m._mock_positions[0] <= LIM1
    assert m._mock_positions[1] <= LIM2


def test_a_genuinely_excessive_goal_is_still_rejected():
    """The tolerance must not become a silent bypass of the limits."""
    m = make()
    with pytest.raises(ValueError, match="Motor 1 goal"):
        m.write_goal_positions(LIM1 + math.radians(1.0), 0.0)
    with pytest.raises(ValueError, match="Motor 2 goal"):
        m.write_goal_positions(0.0, LIM2 + math.radians(1.0))


def test_negative_goals_beyond_tolerance_are_rejected():
    m = make()
    with pytest.raises(ValueError, match="Motor 1 goal"):
        m.write_goal_positions(-math.radians(1.0), 0.0)


def test_the_tolerance_is_smaller_than_the_encoder_step():
    """0.088 deg is the encoder resolution (4096 counts). A tolerance below that
    cannot mask a movement the hardware could even represent."""
    assert math.degrees(MotorController._LIMIT_TOL) < 0.088


def test_a_normal_mid_range_goal_is_untouched():
    m = make()
    goal = math.radians(45.0)
    m.write_goal_positions(goal, goal)
    assert m._mock_positions[0] == pytest.approx(goal)
    assert m._mock_positions[1] == pytest.approx(goal)
