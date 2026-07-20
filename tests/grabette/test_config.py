"""Unit tests for grabette.config hand → angle-sign derivation.

The per-sensor signs turn raw AS5600L readings into the robot-frame convention
(0 = open, positive = closing). Getting the mirror (left/right) mapping wrong
inverts a finger's angle in every recording, so this table is worth locking down.
"""

from grabette.config import Settings


def _settings(**kw):
    # Pass device_id/name explicitly so construction never touches ~/.cache.
    return Settings(device_id="test", device_name="test", **kw)


def test_right_hand_signs():
    """Right hand derives distal=+1, proximal=-1 (both positive on close)."""
    s = _settings(hand="right")
    assert (s.distal_sign, s.proximal_sign) == (+1, -1)


def test_left_hand_signs():
    """Left hand is the mirror: distal=-1, proximal=+1."""
    s = _settings(hand="left")
    assert (s.distal_sign, s.proximal_sign) == (-1, +1)


def test_explicit_signs_override_hand_default():
    """Explicitly set signs win over the hand-derived defaults."""
    s = _settings(hand="right", distal_sign=-1, proximal_sign=+1)
    assert (s.distal_sign, s.proximal_sign) == (-1, +1)
