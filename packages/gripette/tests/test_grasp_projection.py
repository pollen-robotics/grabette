"""Tests for the grasp projection.

The load-bearing property is that encode and decode are exact inverses: the
dataset converter encodes and the eval loop decodes, so any disagreement between
them silently corrupts every grasp with no error anywhere. Each test below maps
to a specific way that could break.
"""

import math

import pytest

from gripette.grasp_projection import (
    GraspProjection,
    _median_filter,
    clamp_to_command_limits,
    normalize_closing_sign,
)


@pytest.mark.parametrize("s", [0.0, 0.15, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("c", [0.0, 0.05, 0.25, 0.5, 0.85, 1.0])
def test_round_trip_is_exact(s, c):
    """decode -> encode must return what we started with, everywhere on the square."""
    gp = GraspProjection()
    prox, dist = gp.decode(s, c)
    s2, c2 = gp.encode(prox, dist)
    assert c2 == pytest.approx(c, abs=1e-9)
    assert s2 == pytest.approx(s, abs=1e-9)


@pytest.mark.parametrize("prox_deg", [0.0, 5.0, 48.5, 93.5])
@pytest.mark.parametrize("dist_deg", [0.0, 13.6, 60.0, 102.0])
def test_round_trip_is_exact_from_the_angle_side(prox_deg, dist_deg):
    """The other direction, on real angle values: the converter starts from
    recorded ANGLES, so encode -> decode must also be the identity."""
    gp = GraspProjection()
    prox, dist = math.radians(prox_deg), math.radians(dist_deg)
    s, c = gp.encode(prox, dist)
    prox2, dist2 = gp.decode(s, c)
    assert prox2 == pytest.approx(prox, abs=1e-12)
    assert dist2 == pytest.approx(dist, abs=1e-12)


def test_channels_are_independent():
    """Closure drives the proximal joint, strategy IS the distal target. Chosen
    over an earlier polar form, which coupled the two and amplified the
    demonstrated distal spread x2.71 (35.8 deg of human variation became 96.8 deg
    of commanded variation)."""
    gp = GraspProjection()
    prox, dist = gp.decode(0.3, 0.7)
    assert prox == pytest.approx(0.7 * gp.lim_prox)
    assert dist == pytest.approx(0.3 * gp.lim_dist)


def test_strategy_does_not_move_the_proximal_joint():
    """Independence, stated as the property that matters: changing the SHAPE must
    not change how far the closing joint is driven."""
    gp = GraspProjection()
    proxes = {gp.decode(s, 0.6)[0] for s in (0.0, 0.25, 0.5, 0.75, 1.0)}
    assert len(proxes) == 1


def test_full_close_always_maxes_the_proximal_joint():
    """The under-close is a PROXIMAL shortfall, so a full close must drive that
    joint to its limit whatever the shape — unlike polar, where a high s pulled
    the commanded proximal DOWN to 37 deg."""
    gp = GraspProjection()
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        prox, dist = gp.full_close(s)
        assert prox == pytest.approx(gp.lim_prox)
        assert dist == pytest.approx(s * gp.lim_dist)


def test_does_not_amplify_the_distal_spread():
    """The measurement that decided this parameterisation: a spread in
    demonstrated distal must produce the SAME spread in the commanded distal."""
    gp = GraspProjection()
    lo_s, _ = gp.encode(0.85, math.radians(2.0))
    hi_s, _ = gp.encode(0.85, math.radians(38.0))
    _p_lo, d_lo = gp.full_close(lo_s)
    _p_hi, d_hi = gp.full_close(hi_s)
    assert d_hi - d_lo == pytest.approx(math.radians(38.0 - 2.0), abs=1e-6)


def test_closure_is_monotone_in_the_proximal_angle():
    """Closure has to be an ordering on "how shut is it" or the segmenter's ramp
    detection is meaningless."""
    gp = GraspProjection()
    cs = [gp.encode(math.radians(d), 0.2)[1] for d in (0, 10, 30, 60, 90)]
    assert cs == sorted(cs)
    assert len(set(cs)) == len(cs)


def test_strategy_is_defined_at_the_open_pose():
    """No singularity: the distal axis is meaningful even when nothing is closed,
    so no nan-filling is needed anywhere downstream."""
    gp = GraspProjection()
    s, c = gp.encode(0.0, 0.0)
    assert not math.isnan(s)
    assert (s, c) == (0.0, 0.0)


def test_inputs_beyond_the_limits_are_clamped():
    """Real recordings exceed the servo range (one set reaches 101% of the
    proximal limit). That must not encode to c > 1."""
    gp = GraspProjection()
    s, c = gp.encode(gp.lim_prox * 1.2, gp.lim_dist * 1.3)
    assert (s, c) == (1.0, 1.0)
    prox, dist = gp.decode(s, c)
    assert prox <= gp.lim_prox + 1e-9
    assert dist <= gp.lim_dist + 1e-9


def test_negative_angles_encode_to_zero_not_to_a_negative_channel():
    """The device's zero is a calibration good to ~1 deg, so the recorded open
    pose sits slightly BELOW zero. Those frames must not produce negative
    channels, which would train the policy on values it can never command."""
    gp = GraspProjection()
    s, c = gp.encode(math.radians(-0.8), math.radians(-0.5))
    assert (s, c) == (0.0, 0.0)


@pytest.mark.parametrize("over", [1.001, 1.02, 1.04, 5.0])
def test_decode_saturates_above_one(over):
    """Measured on the real arm: a trained policy predicts closure up to 1.04.
    That is the model saturating the channel as designed, so decode must treat it
    as a full close rather than commanding past the joint limit."""
    gp = GraspProjection()
    prox, dist = gp.decode(over, over)
    assert prox == pytest.approx(gp.lim_prox)
    assert dist == pytest.approx(gp.lim_dist)


def test_measured_mustard_grasp_encodes_as_expected():
    """Anchor against the real measurement: the mean mustard grasp pose is
    (0.893, 0.296) rad. Normalised on the MEASURED reachable travel (93.5 / 102
    deg), not on the server's looser command bounds."""
    gp = GraspProjection()
    s, c = gp.encode(0.893, 0.296)
    assert c == pytest.approx(0.547, abs=0.005)
    assert s == pytest.approx(0.166, abs=0.005)
    prox_full, _ = gp.full_close(s)
    assert prox_full == pytest.approx(gp.lim_prox, abs=1e-9)
    # The whole point: a full close commands materially more travel.
    assert math.degrees(prox_full - 0.893) == pytest.approx(42.3, abs=1.0)


def test_reachable_travel_is_the_measured_travel():
    """The normaliser must be the measured travel, not a CAD figure. Using a
    bound larger than the real travel understates that channel and biases every
    strategy toward the other joint."""
    gp = GraspProjection()
    assert gp.lim_prox == pytest.approx(math.radians(93.5))
    assert gp.lim_dist == pytest.approx(math.radians(102.0))


def test_a_full_close_is_now_an_acceptable_command():
    """The proximal bound was raised from 85 to the measured 93.5 deg, so a full
    close no longer has to be clamped away. This is the point of raising it: the
    last 8.5 deg is where a firm close lives."""
    from gripette.config import settings
    gp = GraspProjection()
    prox, dist = gp.full_close(0.0)
    cp, _cd, clamped = clamp_to_command_limits(prox, dist)
    assert clamped is False, "a full close must survive the command bounds intact"
    assert cp == pytest.approx(prox)
    assert settings.motor1_max >= gp.lim_prox, "the bound must not cut real travel"


def test_the_clamp_still_guards_against_out_of_range_targets():
    """Kept as a guard: the bound and the reachable travel are separate numbers
    and may drift apart again on another device."""
    from gripette.config import settings
    cp, cd, clamped = clamp_to_command_limits(settings.motor1_max * 2, -1.0)
    assert clamped is True
    assert cp == pytest.approx(settings.motor1_max)
    assert cd == pytest.approx(settings.motor2_min)


def test_clamp_leaves_reachable_targets_alone():
    gp = GraspProjection()
    prox, dist = gp.decode(0.3, 0.5)
    cp, cd, clamped = clamp_to_command_limits(prox, dist)
    assert clamped is False
    assert (cp, cd) == (prox, dist)


def test_negative_closing_convention_is_flipped():
    """Older datasets close negative; encoding them unflipped puts every pose on
    the wrong side of the open pose."""
    vals = [-0.1, -0.5, -0.9, -0.4]
    out, flipped = normalize_closing_sign(vals)
    assert flipped is True
    assert out == [0.1, 0.5, 0.9, 0.4]


def test_positive_convention_is_left_alone():
    vals = [0.1, 0.5, 0.9]
    out, flipped = normalize_closing_sign(vals)
    assert flipped is False
    assert out == vals


def test_an_empty_channel_is_handled():
    """The converter runs over whatever episodes exist, including degenerate ones."""
    out, flipped = normalize_closing_sign([])
    assert (out, flipped) == ([], False)


def test_a_few_glitched_frames_do_not_flip_an_episode():
    """A couple of bad frames in a realistic-length episode must not invert it."""
    vals = [0.4, 0.5, 0.6, 0.5] * 50 + [-9.0, -9.0]
    out, flipped = normalize_closing_sign(vals)
    assert flipped is False
    assert out == vals


def test_a_mostly_open_episode_is_not_flipped_by_noise():
    """The failure this heuristic was rewritten for: an episode that sits near
    zero for most of its length, with a small amount of negative noise, used to
    be judged by its median and mislabelled (51 of 199 real mustard episodes).
    The direction of TRAVEL is what decides it."""
    vals = [-0.001, 0.0, -0.002, 0.001] * 40 + [0.3, 0.6, 0.9, 0.6]
    out, flipped = normalize_closing_sign(vals)
    assert flipped is False, "a positive-closing episode must not be flipped"
    assert out == vals


def test_a_genuinely_negative_episode_is_still_flipped():
    """The mirror case must keep working after that rewrite."""
    vals = [0.001, 0.0, -0.002, 0.001] * 40 + [-0.3, -0.6, -0.9, -0.6]
    out, flipped = normalize_closing_sign(vals)
    assert flipped is True
    assert max(out) == pytest.approx(0.9)


def test_median_filter_kills_a_spike_at_the_very_end():
    """The edge window is SHIFTED INWARD rather than truncated. Truncating leaves
    a 2-3 sample window at the boundary, where a spike PAIR is its own majority
    and survives — and a trailing spike pair is a common recording artifact."""
    vals = [0.5] * 10 + [9.0, 9.0]
    out = _median_filter(vals, width=5)
    assert max(out) == pytest.approx(0.5), "trailing spike pair survived the filter"


def test_median_filter_kills_a_spike_at_the_start():
    vals = [9.0, 9.0] + [0.5] * 10
    out = _median_filter(vals, width=5)
    assert max(out) == pytest.approx(0.5)


def test_median_filter_passes_short_input_through():
    """Shorter than the window: nothing to do, and it must not raise."""
    assert _median_filter([1.0, 2.0], width=5) == [1.0, 2.0]


@pytest.mark.parametrize("bad", [dict(lim_prox=0.0), dict(lim_dist=-1.0),
                                 dict(lim_prox=-0.1)])
def test_invalid_parameters_are_rejected(bad):
    """A silently-wrong projection corrupts data with no error; fail loudly."""
    with pytest.raises(ValueError):
        GraspProjection(**bad)


def test_projection_is_shareable_and_immutable():
    """The converter and the eval loop must be able to share one instance; if it
    were mutable, a drifting limit would desynchronise encode from decode."""
    gp = GraspProjection()
    with pytest.raises(Exception):
        gp.lim_prox = 2.0  # type: ignore[misc]


def test_a_custom_projection_still_round_trips():
    """The limits are the only calibration left, so a device with different
    travel must still encode/decode consistently."""
    gp = GraspProjection(lim_prox=math.radians(80.0), lim_dist=math.radians(95.0))
    prox, dist = gp.decode(0.4, 0.9)
    assert prox == pytest.approx(0.9 * math.radians(80.0))
    assert gp.encode(prox, dist) == pytest.approx((0.4, 0.9))
