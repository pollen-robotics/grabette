"""Unit tests for grabette.hardware.sync.SyncManager (pure clock math).

SyncManager is the shared t=0 reference that places every capture stream (OAK,
Arducam, IMU) on one timeline — the foundation the whole synchronized dataset
rests on. These tests pin its clock conversions without touching real hardware
clocks (the start reference is injected).
"""

import pytest

from grabette.hardware.sync import SyncManager


def test_not_started_raises():
    """Every conversion raises RuntimeError before start() is called."""
    sm = SyncManager()
    assert sm.is_started is False
    with pytest.raises(RuntimeError):
        sm.get_timestamp_ms()
    with pytest.raises(RuntimeError):
        sm.monotonic_s_to_ms(1.0)
    with pytest.raises(RuntimeError):
        sm.boottime_ns_to_ms(1_000_000)


def test_monotonic_s_to_ms_is_relative_to_start():
    """A monotonic-seconds stamp maps to ms elapsed since the start reference."""
    sm = SyncManager()
    sm._start_time = 100.0  # inject a fixed monotonic start (seconds)
    # A stamp 0.25 s after start -> 250 ms on the shared timeline.
    assert sm.monotonic_s_to_ms(100.25) == pytest.approx(250.0)
    assert sm.monotonic_s_to_ms(100.0) == pytest.approx(0.0)


def test_boottime_ns_to_ms_is_relative_to_start():
    """A CLOCK_BOOTTIME-ns stamp lands on the same t=0 ms timeline."""
    sm = SyncManager()
    sm._start_boottime = 5.0  # seconds on CLOCK_BOOTTIME at start
    # 5.1 s expressed in ns -> 100 ms after start.
    assert sm.boottime_ns_to_ms(5_100_000_000) == pytest.approx(100.0)


def test_reset_clears_state():
    """reset() returns the manager to the not-started state."""
    sm = SyncManager()
    sm._start_time = 1.0
    sm._start_boottime = 1.0
    sm.reset()
    assert sm.is_started is False
    assert sm._start_boottime is None
