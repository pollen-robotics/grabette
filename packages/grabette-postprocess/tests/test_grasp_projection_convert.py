"""Tests for the episode conversion that produces the (strategy, closure) dataset.

This is the step that decides what the policy is trained to imitate, and it fails
SILENTLY: a wrong label just teaches the wrong thing, with no error anywhere. The
properties below are the ones the training run depends on.

Trajectories are synthesised rather than loaded so the expected labelling is known
by construction. Shapes follow the real recordings: a long approach at an
intermediate rest posture, a fast closing ramp, then a sustained hold.
"""

import json
import math

import pytest

from gripette.grasp_projection import GraspProjection
from grabette_postprocess.grasp_projection_convert import C_CLOSE, convert_episode

GP = GraspProjection()


def pick_and_lift(rest_c=0.22, grasp_c=0.60, approach=60, ramp=12, hold=40, s=0.15):
    """Angle trajectory for one grasp: approach at rest, ramp, then hold.

    Returns (prox, dist) in radians, built by DECODING a known (s, c) path so the
    conversion's own encode is being checked against an independent construction.
    """
    cs = ([rest_c] * approach
          + [rest_c + (grasp_c - rest_c) * (i + 1) / ramp for i in range(ramp)]
          + [grasp_c] * hold)
    pairs = [GP.decode(s, c) for c in cs]
    return [p for p, _ in pairs], [d for _, d in pairs]


def test_the_close_is_commanded_all_the_way():
    """The whole point of the projection: on close frames the ACTION closure is a
    full close, not the angle the human's fingers happened to stop at."""
    prox, dist = pick_and_lift()
    action, _state, report = convert_episode(prox, dist, GP, rest_is_closed=False)
    assert report["n_close"] == 1
    closes = action[:, 1][action[:, 1] >= C_CLOSE]
    assert len(closes) > 0, "no frame was commanded to a full close"
    assert closes.max() == pytest.approx(1.0)


def test_the_observed_closure_never_reaches_a_full_close():
    """The asymmetry is deliberate and load-bearing: the STATE must keep saying
    where the fingers actually are, because that gap is the only signal for
    "something is in the hand". If the state were rewritten too, the policy could
    not tell a holding gripper from a commanded one."""
    prox, dist = pick_and_lift(grasp_c=0.60)
    _action, state, _report = convert_episode(prox, dist, GP, rest_is_closed=False)
    assert state[:, 1].max() == pytest.approx(0.60, abs=1e-6)
    assert state[:, 1].max() < C_CLOSE


def test_frames_outside_the_grasp_pass_through_untouched():
    """Only the grasp is rewritten. If the approach were altered, the arm would
    be trained to reach with the wrong aperture — measured at 76 deg of error
    when an earlier version zeroed the closure on open frames."""
    prox, dist = pick_and_lift(rest_c=0.22, approach=60)
    action, state, _report = convert_episode(prox, dist, GP, rest_is_closed=False)
    # The early approach is far from the ramp, so it must be identical in both.
    assert action[:40, 1] == pytest.approx(state[:40, 1], abs=1e-6)
    assert action[:40, 1] == pytest.approx(0.22, abs=1e-6)


def test_the_strategy_is_latched_across_the_grasp():
    """`s` drifts while the fingers settle on the object, and since the command is
    decode(s, 1.0) the target then wanders — observed swinging between 57 and 93
    deg of proximal mid-grasp, which makes an already-stalled servo shuffle. A
    grasp has ONE shape."""
    rest, grasp, approach, ramp, hold = 0.22, 0.60, 60, 12, 40
    cs = ([rest] * approach
          + [rest + (grasp - rest) * (i + 1) / ramp for i in range(ramp)]
          + [grasp] * hold)
    # Drifting shape: s wobbles frame to frame during the hold.
    ss = [0.15] * (approach + ramp) + [0.15 + 0.04 * math.sin(i) for i in range(hold)]
    pairs = [GP.decode(s, c) for s, c in zip(ss, cs)]
    prox = [p for p, _ in pairs]
    dist = [d for _, d in pairs]

    action, state, report = convert_episode(prox, dist, GP, rest_is_closed=False)
    assert report["n_close"] == 1
    closing = [i for i, c in enumerate(action[:, 1]) if c >= C_CLOSE]
    latched = {round(float(action[i, 0]), 9) for i in closing}
    assert len(latched) == 1, f"strategy wandered across the grasp: {sorted(latched)}"
    # The state must still carry the real, drifting shape.
    assert len({round(float(state[i, 0]), 6) for i in closing}) > 1


def test_a_negative_convention_episode_is_flipped_and_still_converts():
    """A mixed corpus reaches the converter; an unflipped episode would encode
    every pose on the wrong side of the open pose and produce no close at all."""
    prox, dist = pick_and_lift()
    action, _state, report = convert_episode(
        [-p for p in prox], [-d for d in dist], GP, rest_is_closed=False
    )
    assert report["sign_flipped"] is True
    assert action[:, 1].max() == pytest.approx(1.0)


def test_a_flat_episode_produces_no_close():
    """No grasp happened, so nothing may be rewritten — inventing a close here
    would teach the policy to shut on nothing."""
    prox, dist = zip(*[GP.decode(0.15, 0.22) for _ in range(120)])
    action, state, report = convert_episode(
        list(prox), list(dist), GP, rest_is_closed=False
    )
    assert report["n_close"] == 0
    assert report["frac_close"] == 0.0
    assert action[:, 1] == pytest.approx(state[:, 1], abs=1e-6)


def test_the_reported_close_fraction_matches_the_rewritten_frames():
    """The report is what the operator judges a conversion by, so it has to agree
    with what was actually written."""
    prox, dist = pick_and_lift()
    action, _state, report = convert_episode(prox, dist, GP, rest_is_closed=False)
    rewritten = sum(1 for c in action[:, 1] if c >= C_CLOSE)
    assert report["frac_close"] == pytest.approx(rewritten / len(action), abs=1e-9)


def test_rest_closed_convention_commands_a_close_at_rest_too():
    """Under the rest-closed convention idle and grasp are the SAME command, so
    everything outside a deliberate open is a full close. That is what removes the
    release-vs-wide-grasp ambiguity."""
    # Closed at rest, a deliberate open in the middle, then closed again.
    cs = [0.55] * 40 + [0.05] * 30 + [0.55] * 40
    pairs = [GP.decode(0.15, c) for c in cs]
    prox = [p for p, _ in pairs]
    dist = [d for _, d in pairs]
    action, _state, _report = convert_episode(prox, dist, GP, rest_is_closed=True)
    assert action[0, 1] == pytest.approx(1.0), "idle must be a commanded close"
    assert action[-1, 1] == pytest.approx(1.0)
    # The open window must NOT be commanded closed.
    assert min(action[:, 1]) < C_CLOSE


def test_output_is_float32_pairs():
    """The dataset writer expects exactly two float32 channels per frame; a dtype
    or shape drift here corrupts the parquet silently."""
    prox, dist = pick_and_lift()
    action, state, _report = convert_episode(prox, dist, GP, rest_is_closed=False)
    assert action.dtype.name == "float32"
    assert state.dtype.name == "float32"
    assert action.shape == state.shape == (len(prox), 2)


def test_channels_stay_inside_the_trained_range():
    """Both channels are normalised, so nothing may leave [0, 1] — an out-of-range
    value would be unnormalisable at training time and unreachable at eval."""
    prox, dist = pick_and_lift()
    action, state, _report = convert_episode(prox, dist, GP, rest_is_closed=False)
    for arr in (action, state):
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0


# ── what a converted dataset records about itself ───────────────────────

def test_the_sidecar_records_the_calibration_that_built_the_dataset(tmp_path):
    """Channel names say WHICH representation; only this says which CALIBRATION,
    and decode has to use the same limits encode did."""
    from gripette.grasp_projection import PROJECTION_SIDECAR
    from grabette_postprocess.grasp_projection_convert import _write_sidecar

    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    _write_sidecar(root, GP, tmp_path / "src")

    meta = json.loads((root / PROJECTION_SIDECAR).read_text())
    assert GP.matches_metadata(meta) is None
    assert meta["source_root"].endswith("src")


def test_a_drifted_projection_no_longer_matches_its_own_sidecar(tmp_path):
    """The guard has to actually fire: re-measured travel must not silently pass."""
    from gripette.grasp_projection import GraspProjection

    meta = GP.to_metadata()
    drifted = GraspProjection(lim_prox=GP.lim_prox, lim_dist=GP.lim_dist + 0.05)
    assert drifted.matches_metadata(meta) is not None


def test_the_card_carries_the_tags_and_the_badge(tmp_path):
    from grabette_postprocess.grasp_projection_convert import _write_card

    root = tmp_path / "ds"
    root.mkdir()
    _write_card(root, repo_id="user/ds")
    card = (root / "README.md").read_text()
    for tag in ("LeRobot", "grasp-projection"):
        assert f"- {tag}" in card, tag
    assert "visualize_dataset?path=user/ds" in card


def test_the_card_is_written_without_a_badge_when_the_repo_is_unknown(tmp_path):
    """A local conversion has no repo id yet; the tags are still worth writing."""
    from grabette_postprocess.grasp_projection_convert import _write_card

    root = tmp_path / "ds"
    root.mkdir()
    _write_card(root)
    card = (root / "README.md").read_text()
    assert "- grasp-projection" in card
    assert "visualize_dataset" not in card


def test_an_existing_card_is_never_overwritten(tmp_path):
    """A hand-written card carries more than the generated one can."""
    from grabette_postprocess.grasp_projection_convert import _write_card

    root = tmp_path / "ds"
    root.mkdir()
    (root / "README.md").write_text("hand written, do not clobber\n")
    _write_card(root, repo_id="user/ds")
    assert (root / "README.md").read_text() == "hand written, do not clobber\n"
