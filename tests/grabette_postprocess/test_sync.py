"""Unit tests for grabette_postprocess.checks.sync pure numerics."""

import numpy as np

from grabette_postprocess.checks.sync import classify_lag, cross_correlate_signals


def test_classify_lag_verdicts():
    """Lag magnitude maps to GOOD (<20ms) / MARGINAL (20-50ms) / BAD (>50ms), sign-agnostic."""
    assert classify_lag(0.005, 0.9)[0] == "GOOD"       # < 20 ms
    assert classify_lag(0.030, 0.9)[0] == "MARGINAL"   # 20-50 ms
    assert classify_lag(0.080, 0.9)[0] == "BAD"        # > 50 ms
    # Sign doesn't matter, magnitude does.
    assert classify_lag(-0.080, 0.9)[0] == "BAD"


def test_classify_lag_clean_good_has_no_note():
    """A good lag with strong correlation produces an empty note."""
    verdict, note = classify_lag(0.001, 0.95)
    assert verdict == "GOOD" and note == ""


def test_classify_lag_low_correlation_adds_caveat():
    """A weak correlation appends a caveat to the note even when the lag is good."""
    verdict, note = classify_lag(0.001, 0.1)   # good lag but weak correlation
    assert verdict == "GOOD"
    assert "low correlation" in note


def test_cross_correlate_recovers_known_offset():
    """Cross-correlation recovers a known time offset between two signals."""
    # Use a single Gaussian pulse (non-periodic) so the recovered lag is
    # unambiguous — a sine's period makes the sign ambiguous. s2's pulse sits
    # 0.1 s later than s1's; the implementation reports that as lag = -0.1.
    t = np.arange(0.0, 5.0, 0.005)
    pulse = lambda center: np.exp(-((t - center) ** 2) / (2 * 0.05 ** 2))
    best_lag, best_corr, _, _ = cross_correlate_signals(
        t, pulse(2.0), t, pulse(2.1), max_lag_s=0.5)
    np.testing.assert_allclose(best_lag, -0.1, atol=0.01)
    assert best_corr > 0.9


def test_cross_correlate_no_overlap_returns_zeros():
    """Signals with disjoint time windows return a zero lag/correlation."""
    t1 = np.array([0.0, 1.0])
    t2 = np.array([2.0, 3.0])   # disjoint windows
    best_lag, best_corr, lags, corr = cross_correlate_signals(
        t1, np.array([0.0, 1.0]), t2, np.array([0.0, 1.0]),
    )
    assert best_lag == 0.0 and best_corr == 0.0
