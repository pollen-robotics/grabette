"""Unit tests for grabette_postprocess.convert pure helpers (no ffmpeg / no I/O)."""

import numpy as np

from grabette_postprocess.convert import _ms_to_ns, fit_device_to_host_s


def test_ms_to_ns_rounds_to_nearest_integer():
    """Millisecond→nanosecond conversion rounds to the nearest integer ns."""
    assert _ms_to_ns(1.5) == 1_500_000
    assert _ms_to_ns(0.001) == 1_000          # 1 microsecond
    assert _ms_to_ns(0.0000004) == 0          # rounds down
    assert _ms_to_ns(0.0000006) == 1          # rounds up


def test_fit_device_to_host_recovers_known_affine():
    """The least-squares fit recovers a known device→host slope and intercept."""
    # host_s = 1.0 * device_s + 0.5, expressed in the file's us / ms units.
    samples = [
        {"device_us": 0, "host_ms": 500},        # device_s=0.0 -> host_s=0.5
        {"device_us": 1_000_000, "host_ms": 1500},  # device_s=1.0 -> host_s=1.5
        {"device_us": 2_000_000, "host_ms": 2500},  # device_s=2.0 -> host_s=2.5
    ]
    slope, intercept = fit_device_to_host_s(samples)
    assert isinstance(slope, float) and isinstance(intercept, float)
    np.testing.assert_allclose(slope, 1.0, atol=1e-9)
    np.testing.assert_allclose(intercept, 0.5, atol=1e-9)


def test_fit_device_to_host_none_without_device_clock():
    """Samples lacking device_us return None (legacy recordings fall back to host_ms)."""
    # Legacy recordings carry no device_us -> caller falls back to raw host_ms.
    assert fit_device_to_host_s([{"host_ms": 100}, {"host_ms": 200}]) is None


def test_fit_device_to_host_none_with_fewer_than_two_points():
    """Fewer than two usable points can't define a line -> None."""
    assert fit_device_to_host_s([{"device_us": 0, "host_ms": 0}]) is None
