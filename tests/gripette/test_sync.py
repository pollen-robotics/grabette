"""Unit tests for gripette.hardware.sync.SyncManager."""

import pytest

from gripette.hardware.sync import SyncManager


def test_not_started_raises():
    """get_timestamp_ms raises before start() is called."""
    sm = SyncManager()
    assert sm.is_started is False
    with pytest.raises(RuntimeError):
        sm.get_timestamp_ms()


def test_start_makes_it_started_and_reset_clears():
    """start() begins the clock (first stamp ≥ 0); reset() returns to not-started."""
    sm = SyncManager()
    sm.start()
    assert sm.is_started is True
    # First stamp after start is small and non-negative.
    assert sm.get_timestamp_ms() >= 0.0
    sm.reset()
    assert sm.is_started is False
