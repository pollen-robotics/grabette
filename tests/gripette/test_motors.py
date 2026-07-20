"""Unit tests for gripette.hardware.motors.MotorController — limits + frame math.

The gripper's safety limits and the robot↔encoder conversion are the two pieces
of logic that stand between an RPC command and the physical servos. Both run
without opening the serial bus (no start(), no bus thread), so they're testable
directly. The mock/non-mock path is forced explicitly so the test is independent
of whether rustypot happens to be installed on the runner.
"""

import pytest

from gripette.hardware.motors import MotorController


def _controller(**kw):
    # Never call start() -> no serial, no bus thread.
    return MotorController(**kw)


# ---- limit checking -------------------------------------------------------

def test_check_limits_accepts_in_range():
    """In-range positions pass _check_limits without raising."""
    mc = _controller(limits=((0.0, 1.5), (0.0, 2.0)))
    mc._check_limits(0.5, 1.0)  # should not raise


def test_check_limits_rejects_motor1_over_max():
    """Motor 1 above its max raises ValueError naming motor 1."""
    mc = _controller(limits=((0.0, 1.5), (0.0, 2.0)))
    with pytest.raises(ValueError, match="Motor 1"):
        mc._check_limits(1.6, 1.0)


def test_check_limits_rejects_motor2_under_min():
    """Motor 2 below its min raises ValueError naming motor 2."""
    mc = _controller(limits=((0.0, 1.5), (0.0, 2.0)))
    with pytest.raises(ValueError, match="Motor 2"):
        mc._check_limits(0.5, -0.1)


def test_no_limits_configured_allows_anything():
    """With limits=None, _check_limits accepts any position."""
    mc = _controller(limits=None)
    mc._check_limits(1e6, -1e6)  # should not raise


def test_write_goal_enforces_limits():
    """write_goal_positions rejects an out-of-limit goal before queuing it."""
    mc = _controller(limits=((0.0, 1.5), (0.0, 2.0)))
    mc._mock = True
    with pytest.raises(ValueError):
        mc.write_goal_positions(2.0, 0.0)


# ---- robot <-> encoder conversion (non-mock path, no serial) --------------

def test_write_goal_converts_robot_to_encoder_frame():
    """Non-mock write converts robot→encoder (encoder = robot*sign + offset) into the pending slot."""
    # encoder = robot * sign + offset
    mc = _controller(signs=(-1, 1), offsets=(0.5, 0.0), limits=None)
    mc._mock = False  # exercise the conversion, but never start the bus thread
    mc.write_goal_positions(0.2, 0.3)
    assert mc._pending_goal == pytest.approx((0.2 * -1 + 0.5, 0.3 * 1 + 0.0))


# ---- mock round-trip ------------------------------------------------------

def test_mock_write_then_read_is_robot_frame_identity():
    """In mock mode, read_positions returns exactly the robot-frame goal written."""
    mc = _controller(signs=(-1, 1), offsets=(0.5, 0.0), limits=None)
    mc._mock = True
    # Mock keeps robot-frame state (no encoder conversion), matching read semantics.
    mc.write_goal_positions(0.2, 0.3)
    assert mc.read_positions() == pytest.approx((0.2, 0.3))
