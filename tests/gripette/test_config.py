"""Unit tests for gripette.config: hand → motor-sign mapping and limit defaults."""

import math

from gripette.config import Settings


def test_right_hand_motor_signs():
    """Right hand derives both motor signs as +1."""
    s = Settings(hand="right")
    assert (s.motor1_sign, s.motor2_sign) == (+1, +1)


def test_left_hand_motor_signs():
    """Left hand is the mirror: both motor signs -1."""
    s = Settings(hand="left")
    assert (s.motor1_sign, s.motor2_sign) == (-1, -1)


def test_explicit_signs_override_hand():
    """Explicitly set motor signs win over the hand-derived defaults."""
    s = Settings(hand="left", motor1_sign=+1, motor2_sign=-1)
    assert (s.motor1_sign, s.motor2_sign) == (+1, -1)


def test_default_robot_frame_limits():
    """Default limits are 0 (open) to the mechanical closing angles in radians."""
    s = Settings()
    # 0 = fully open; max in radians from the mechanical closing angles.
    assert s.motor1_min == 0.0 and s.motor2_min == 0.0
    assert s.motor1_max == math.radians(85)
    assert s.motor2_max == math.radians(116)
