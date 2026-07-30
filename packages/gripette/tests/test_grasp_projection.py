"""Tests for the grasp projection.

The load-bearing property is that encode and decode are exact inverses: the
dataset converter encodes and the eval loop decodes, so any disagreement between
them silently corrupts every grasp with no error anywhere. Each test below maps
to a specific way that could break.
"""

import math

import pytest

from gripette.grasp_projection import GraspProjection, normalize_closing_sign

# The parameter sets the projection must work under: the plain geometric default,
# a curved boundary, a leading-proximal path, and both refinements at once.
PARAM_SETS = [
    dict(),                                  # defaults: box boundary, straight path
    dict(p=2.0),                             # quarter-circle boundary
    dict(p=3.0),                             # superellipse between the two
    dict(a=1.0, b=1.6),                      # distal curls late
    dict(a=0.8, b=1.5),                      # both exponents off 1
    dict(p=2.5, a=0.9, b=1.4),               # everything at once
]


@pytest.mark.parametrize("params", PARAM_SETS)
@pytest.mark.parametrize("s", [0.0, 0.15, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("c", [0.05, 0.25, 0.5, 0.85, 1.0])
def test_round_trip_is_exact(params, s, c):
    """decode -> encode must return what we started with, for every parameter set."""
    gp = GraspProjection(**params)
    prox, dist = gp.decode(s, c)
    s2, c2 = gp.encode(prox, dist)
    assert c2 == pytest.approx(c, abs=1e-6)
    # At s = 0 or 1 one joint is exactly zero, which pins the direction; in
    # between, bisection tolerance applies.
    assert s2 == pytest.approx(s, abs=1e-5)


@pytest.mark.parametrize("params", PARAM_SETS)
def test_full_close_reaches_the_boundary(params):
    """c = 1 must sit ON the boundary — that is what makes 'fully closed' mean
    something. If it fell short, the under-close bug would survive the rewrite."""
    gp = GraspProjection(**params)
    for s in (0.0, 0.2, 0.5, 0.9, 1.0):
        u, v = gp.boundary(s)
        if math.isinf(gp.p):
            assert max(u, v) == pytest.approx(1.0, abs=1e-9)
        else:
            assert u ** gp.p + v ** gp.p == pytest.approx(1.0, abs=1e-9)


def test_default_boundary_is_the_box_corner():
    """With p = inf the joints reach their limits independently."""
    gp = GraspProjection()
    assert gp.boundary(0.0) == pytest.approx((1.0, 0.0), abs=1e-9)
    assert gp.boundary(1.0) == pytest.approx((0.0, 1.0), abs=1e-9)
    assert gp.boundary(0.5) == pytest.approx((1.0, 1.0), abs=1e-9)  # both maxed


def test_circle_boundary_trades_the_joints_off():
    """p = 2: closing hard on one joint costs range on the other — the physical
    behaviour of a finger that fouls the thumb."""
    gp = GraspProjection(p=2.0)
    u, v = gp.boundary(0.5)
    assert u == pytest.approx(math.sqrt(0.5), abs=1e-9)
    assert v == pytest.approx(math.sqrt(0.5), abs=1e-9)
    assert u < 1.0 and v < 1.0


def test_strategy_axis_orientation():
    """s = 0 must be pure PROXIMAL and s = 1 pure DISTAL.

    Pinned deliberately: this axis was inverted once during design, which made
    every conclusion about grasp shape backwards.
    """
    gp = GraspProjection()
    prox0, dist0 = gp.full_close(0.0)
    prox1, dist1 = gp.full_close(1.0)
    assert prox0 == pytest.approx(gp.lim_prox) and dist0 == pytest.approx(0.0)
    assert dist1 == pytest.approx(gp.lim_dist) and prox1 == pytest.approx(0.0)


def test_closure_is_monotone_in_the_angles():
    """Closing further must never read as less closed."""
    gp = GraspProjection()
    prev = -1.0
    for frac in [i / 20 for i in range(21)]:
        prox, dist = gp.decode(0.3, frac)
        _s, c = gp.encode(prox, dist)
        assert c >= prev - 1e-9
        prev = c


def test_open_pose_has_undefined_strategy():
    """Both joints at zero carry no shape information; say so instead of
    inventing a value."""
    gp = GraspProjection()
    s, c = gp.encode(0.0, 0.0)
    assert math.isnan(s)
    assert c == 0.0


def test_inputs_beyond_the_limits_are_clamped():
    """Real recordings exceed the servo range (one set reaches 101% of the
    proximal limit). That must not encode to c > 1."""
    gp = GraspProjection()
    s, c = gp.encode(gp.lim_prox * 1.2, gp.lim_dist * 1.3)
    assert 0.0 <= c <= 1.0
    assert 0.0 <= s <= 1.0
    prox, dist = gp.decode(s, c)
    assert prox <= gp.lim_prox + 1e-9
    assert dist <= gp.lim_dist + 1e-9


def test_measured_mustard_grasp_encodes_as_expected():
    """Anchor against the real measurement: the mean mustard grasp pose was
    (0.893, 0.296) rad, which step 0 found to be 60% of the proximal range. The
    projection must agree, and a full close must recover the unused travel."""
    gp = GraspProjection()
    s, c = gp.encode(0.893, 0.296)
    assert s == pytest.approx(0.152, abs=0.005)
    assert c == pytest.approx(0.602, abs=0.005)
    prox_full, _dist_full = gp.full_close(s)
    assert prox_full == pytest.approx(gp.lim_prox, abs=1e-9)
    # The whole point: a full close commands materially more travel.
    assert math.degrees(prox_full - 0.893) == pytest.approx(34.0, abs=1.0)


def test_encode_trajectory_fills_the_open_frames():
    """Leading and trailing open frames must not leak nan into a dataset."""
    gp = GraspProjection()
    mid_prox, mid_dist = gp.decode(0.4, 0.8)
    prox = [0.0, 0.0, mid_prox, mid_prox, 0.0]
    dist = [0.0, 0.0, mid_dist, mid_dist, 0.0]
    s, c, closed = gp.encode_trajectory(prox, dist, close_at=0.5)
    assert not any(math.isnan(v) for v in s)
    assert all(v == pytest.approx(0.4, abs=1e-5) for v in s)
    assert closed == [False, False, True, True, False]
    assert c[0] == 0.0


def test_encode_trajectory_rejects_mismatched_lengths():
    gp = GraspProjection()
    with pytest.raises(ValueError, match="length mismatch"):
        gp.encode_trajectory([0.1, 0.2], [0.1])


def test_close_threshold_moves_the_onset():
    """The threshold is the tunable knob; confirm it actually shifts the onset,
    so a sweep over it is meaningful."""
    gp = GraspProjection()
    prox, dist = zip(*[gp.decode(0.3, i / 10) for i in range(11)])
    _s, _c, early = gp.encode_trajectory(list(prox), list(dist), close_at=0.3)
    _s, _c, late = gp.encode_trajectory(list(prox), list(dist), close_at=0.8)
    assert early.index(True) < late.index(True)


def test_negative_closing_convention_is_flipped():
    """Older datasets close negative; encoding them unflipped puts every pose in
    the wrong quadrant."""
    vals = [-0.1, -0.5, -0.9, -0.4]
    out, flipped = normalize_closing_sign(vals)
    assert flipped is True
    assert out == [0.1, 0.5, 0.9, 0.4]


def test_positive_convention_is_left_alone():
    vals = [0.1, 0.5, 0.9]
    out, flipped = normalize_closing_sign(vals)
    assert flipped is False
    assert out == vals


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


@pytest.mark.parametrize("bad", [dict(p=0.5), dict(a=0.0), dict(b=-1.0),
                                 dict(lim_prox=0.0), dict(lim_dist=-1.0)])
def test_invalid_parameters_are_rejected(bad):
    """A silently-wrong projection corrupts data with no error; fail loudly."""
    with pytest.raises(ValueError):
        GraspProjection(**bad)


def test_projection_is_shareable_and_immutable():
    """The converter and the eval loop must be able to share one instance; if it
    were mutable, a drifting parameter would desynchronise encode from decode."""
    gp = GraspProjection()
    with pytest.raises(Exception):
        gp.p = 2.0  # type: ignore[misc]
