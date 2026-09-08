"""Tests for chunk-relative actions.

The load-bearing property is ORIGIN INVARIANCE: SLAM gives every episode an
arbitrary origin, so if re-originating an episode changes the targets, the
representation is unusable. That test also verifies it has teeth, by checking the
naive alternative (elementwise subtraction) FAILS it.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from grabette_chunkrel.chunk_relative import (
    DEFAULT_JUMP_CAP_DEG,
    DEFAULT_JUMP_CAP_M,
    from_chunk_relative,
    splice_jumps,
    splice_rotation_jumps,
    to_chunk_relative,
)

rng = np.random.default_rng(0)


def make_chunk(t=20, jitter=0.02, seed=0):
    """A plausible absolute-pose chunk: smooth drift from an arbitrary origin."""
    r = np.random.default_rng(seed)
    pos = np.cumsum(r.normal(scale=jitter, size=(t, 3)), axis=0) + np.array([3.0, -1.0, 0.7])
    rot = np.cumsum(r.normal(scale=0.03, size=(t, 3)), axis=0) + np.array([0.4, -2.0, 1.1])
    grip = np.abs(r.normal(scale=0.2, size=(t, 2)))
    return np.concatenate([pos, rot, grip], axis=1)


def reorigin(chunk, R_g, t_g):
    """What a different SLAM session gives: a rigid transform of the whole episode."""
    out = chunk.copy()
    out[:, :3] = R_g.apply(chunk[:, :3]) + t_g
    out[:, 3:6] = (R_g * Rotation.from_rotvec(chunk[:, 3:6])).as_rotvec()
    return out


# ── the property the design exists for ──────────────────────────────────

@pytest.mark.parametrize("seed", range(5))
def test_origin_invariance_under_arbitrary_reorigin(seed):
    """A random rigid re-origin must not change a single output value."""
    chunk = make_chunk(seed=seed)
    R_g = Rotation.random(random_state=seed + 100)
    t_g = rng.normal(scale=10.0, size=3)

    a, _ = to_chunk_relative(chunk)
    b, _ = to_chunk_relative(reorigin(chunk, R_g, t_g))
    assert np.abs(a - b).max() < 1e-5, "targets moved with the SLAM origin"


@pytest.mark.parametrize("yaw_deg", [10, 45, 90, 180])
def test_origin_invariance_under_yaw(yaw_deg):
    """The realistic case: trajectories are gravity-aligned, so yaw is the free
    parameter. This is exactly what breaks elementwise subtraction."""
    chunk = make_chunk()
    R_g = Rotation.from_rotvec(np.deg2rad(yaw_deg) * np.array([0, 0, 1.0]))
    a, _ = to_chunk_relative(chunk)
    b, _ = to_chunk_relative(reorigin(chunk, R_g, np.array([5.0, -3.0, 0.0])))
    assert np.abs(a - b).max() < 1e-5


def test_the_invariance_test_has_teeth():
    """Guard against a vacuous test: the naive alternative must FAIL it.

    `p_i - p_ref` without composing into the reference frame is what LeRobot's
    built-in step computes. If that also passed, the test above would be proving
    nothing.
    """
    chunk = make_chunk()
    R_g = Rotation.from_rotvec(np.deg2rad(90) * np.array([0, 0, 1.0]))
    naive = lambda c: c[:, :3] - c[:, :3][0]        # noqa: E731
    drift = np.abs(naive(chunk) - naive(reorigin(chunk, R_g, np.zeros(3)))).max()
    assert drift > 0.01, "the naive form should be origin-DEPENDENT by >1 cm"


# ── round trip ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("rot", ["rotvec", "r6d"])
@pytest.mark.parametrize("seed", range(3))
def test_round_trip_is_exact(rot, seed):
    """The inverse runs at inference to rebuild absolute targets; any error here
    is a systematic offset applied to every commanded pose."""
    chunk = make_chunk(seed=seed)
    rel, _ = to_chunk_relative(chunk, rot=rot)
    back = from_chunk_relative(rel, chunk[0, :6], rot=rot)
    assert np.abs(back[:, :3] - chunk[:, :3]).max() < 1e-5
    # rotations compare as rotations, not as coordinates
    err = (Rotation.from_rotvec(back[:, 3:6]).inv()
           * Rotation.from_rotvec(chunk[:, 3:6])).magnitude()
    assert np.rad2deg(err).max() < 1e-3
    assert np.abs(back[:, 6:] - chunk[:, 6:]).max() < 1e-5


def test_the_first_action_is_the_identity_offset():
    chunk = make_chunk()
    rel, _ = to_chunk_relative(chunk)
    assert np.abs(rel[0, :3]).max() < 1e-6
    assert np.abs(rel[0, 3:6]).max() < 1e-6


# ── rotation must be composed, not subtracted ───────────────────────────

def test_rotation_is_composition_not_subtraction():
    """R_ref^-1 R_i, verified against an independent construction."""
    chunk = make_chunk()
    rel, _ = to_chunk_relative(chunk)
    rots = Rotation.from_rotvec(chunk[:, 3:6])
    want = (rots[0].inv() * rots).as_rotvec()
    assert np.abs(rel[:, 3:6] - want).max() < 1e-5

    # and it is NOT the subtraction, which is the bug we are avoiding
    sub = chunk[:, 3:6] - chunk[0, 3:6]
    assert np.abs(rel[:, 3:6] - sub).max() > 1e-3


def test_relative_rotations_stay_small_enough_for_rotvec():
    """Why rotvec is safe here: within a chunk the relative rotation is far from
    the pi wraparound where axis-angle becomes ambiguous."""
    _rel, rep = to_chunk_relative(make_chunk(t=50))
    assert rep["max_rot_deg"] < 150.0


# ── gripper ─────────────────────────────────────────────────────────────

def test_gripper_channels_pass_through_untouched():
    """A relative closure is meaningless: with the grasp projection, 1.0 means
    'fully closed', an absolute command by construction."""
    chunk = make_chunk()
    rel, _ = to_chunk_relative(chunk)
    assert np.abs(rel[:, 6:] - chunk[:, 6:]).max() < 1e-6


# ── relocalisation jumps ────────────────────────────────────────────────

def test_a_jump_is_spliced_out():
    """A chunk-wide reference spreads a jump across every later action, so it is
    repaired rather than left in. Masking is unavailable: pi0/pi05 ignore
    action_is_pad, and a processor step cannot drop samples."""
    chunk = make_chunk(t=12, jitter=0.005)
    clean, _ = to_chunk_relative(chunk)

    jumped = chunk.copy()
    jumped[6:, :3] += np.array([0.30, -0.10, 0.05])     # 32 cm re-acquisition
    repaired, rep = to_chunk_relative(jumped)

    assert rep["n_jumps_spliced"] == 1
    # NOT exact, and cannot be: the observed jump step is `true_motion + artifact`
    # and the two are inseparable, so the motion during that one step is lost. The
    # contract is that the 32 cm artifact is gone and the residual is jitter-scale.
    step = np.linalg.norm(np.diff(chunk[:, :3], axis=0), axis=1).mean()
    resid = np.abs(repaired - clean).max()
    assert resid < 3 * step, f"residual {resid:.4f} m too large for a {step:.4f} m step"
    assert resid < 0.01, "the artifact itself must be gone"


def test_several_jumps_accumulate_correctly():
    chunk = make_chunk(t=20, jitter=0.005)
    clean, _ = to_chunk_relative(chunk)
    jumped = chunk.copy()
    jumped[5:, :3] += np.array([0.2, 0.0, 0.0])
    jumped[13:, :3] += np.array([0.0, -0.4, 0.1])
    repaired, rep = to_chunk_relative(jumped)
    assert rep["n_jumps_spliced"] == 2
    step = np.linalg.norm(np.diff(chunk[:, :3], axis=0), axis=1).mean()
    # Two jumps totalling 65 cm; the residual must stay jitter-scale, i.e. the
    # errors must not accumulate across jumps.
    assert np.abs(repaired - clean).max() < 6 * step


def test_real_motion_below_the_cap_is_not_spliced():
    """The cap must not eat fast hand motion: 7 cm in one step is under it."""
    pos = np.zeros((5, 3))
    pos[2:, 0] += 0.07
    out, n = splice_jumps(pos, DEFAULT_JUMP_CAP_M)
    assert n == 0
    assert np.abs(out - pos).max() < 1e-12


def test_splice_handles_degenerate_input():
    for pos in (np.zeros((0, 3)), np.zeros((1, 3))):
        out, n = splice_jumps(pos)
        assert n == 0 and out.shape == pos.shape


# ── contracts ───────────────────────────────────────────────────────────

def test_bad_shapes_and_modes_are_rejected():
    with pytest.raises(ValueError, match="expected"):
        to_chunk_relative(np.zeros((10, 7)))
    with pytest.raises(ValueError, match="rot must be"):
        to_chunk_relative(np.zeros((10, 8)), rot="quat")
    with pytest.raises(ValueError, match="expected"):
        from_chunk_relative(np.zeros((10, 8)), np.zeros(3))
    with pytest.raises(ValueError, match="expected"):
        from_chunk_relative(np.zeros((10, 11)), np.zeros(6), rot="rotvec")


def test_output_is_float32():
    """The dataloader path expects float32; a silent float64 doubles batch memory."""
    rel, _ = to_chunk_relative(make_chunk())
    assert rel.dtype == np.float32
    assert from_chunk_relative(rel, make_chunk()[0, :6]).dtype == np.float32


def test_offsets_are_orders_of_magnitude_larger_than_per_step_deltas():
    """The reason to do any of this: supervision SNR. Per-step deltas are mm
    against mm-scale SLAM jitter; chunk offsets are cm against the same noise."""
    chunk = make_chunk(t=50, jitter=0.004)
    rel, rep = to_chunk_relative(chunk)
    per_step = np.linalg.norm(np.diff(chunk[:, :3], axis=0), axis=1).mean()
    assert rep["max_offset_m"] > 8 * per_step


def test_the_repaired_sequence_is_continuous():
    """The point of splicing: no step above the cap survives, so a chunk-wide
    reference stays valid for every action after the jump."""
    chunk = make_chunk(t=15, jitter=0.005)
    chunk[7:, :3] += np.array([0.5, 0.2, -0.3])
    pos, n = splice_jumps(chunk[:, :3])
    assert n == 1
    assert np.linalg.norm(np.diff(pos, axis=0), axis=1).max() < DEFAULT_JUMP_CAP_M


# ── rotation jumps ──────────────────────────────────────────────────────

def test_a_rotation_jump_is_spliced():
    """Rotation glitches are real in this data — convert_dataset.py's
    --despike_max_deg was lowered from 45 to 5 because 5-45 deg glitches reached
    training, were reproduced at eval, and tripped the arm's IK-jump watchdog. So
    repairing translation alone would be a REGRESSION against the delta pipeline.
    """
    chunk = make_chunk(t=14, jitter=0.005)
    clean, _ = to_chunk_relative(chunk)

    jumped = chunk.copy()
    extra = Rotation.from_rotvec(np.deg2rad(40) * np.array([0.0, 1.0, 0.0]))
    jumped[7:, 3:6] = (extra * Rotation.from_rotvec(chunk[7:, 3:6])).as_rotvec()
    repaired, rep = to_chunk_relative(jumped)

    assert rep["n_rot_jumps_spliced"] == 1
    # Residual is bounded by the lost per-step motion, not zero — same
    # information-theoretic limit as the translation case.
    step_deg = np.rad2deg((Rotation.from_rotvec(chunk[:-1, 3:6]).inv()
                           * Rotation.from_rotvec(chunk[1:, 3:6])).magnitude()).mean()
    err_deg = np.rad2deg((Rotation.from_rotvec(repaired[:, 3:6]).inv()
                          * Rotation.from_rotvec(clean[:, 3:6])).magnitude()).max()
    assert err_deg < 3 * step_deg, f"residual {err_deg:.2f}deg vs step {step_deg:.2f}deg"
    assert err_deg < 5.0, "the 40 deg artifact must be gone"


def test_the_repaired_rotation_sequence_is_continuous():
    chunk = make_chunk(t=15, jitter=0.005)
    extra = Rotation.from_rotvec(np.deg2rad(60) * np.array([1.0, 0.0, 0.0]))
    chunk[9:, 3:6] = (extra * Rotation.from_rotvec(chunk[9:, 3:6])).as_rotvec()
    rots, n = splice_rotation_jumps(Rotation.from_rotvec(chunk[:, 3:6]))
    assert n == 1
    steps = np.rad2deg((rots[:-1].inv() * rots[1:]).magnitude())
    assert steps.max() < DEFAULT_JUMP_CAP_DEG + 1e-6


def test_normal_wrist_motion_is_not_spliced():
    """The cap must not eat real motion: 4 deg/step is 200 deg/s at 50 fps, fast
    but human. Only above 5 deg is it treated as a glitch."""
    rots = Rotation.from_rotvec(
        np.cumsum(np.full((10, 3), np.deg2rad(4.0) / np.sqrt(3)), axis=0)
    )
    _out, n = splice_rotation_jumps(rots)
    assert n == 0


def test_rotation_splice_handles_degenerate_input():
    for t in (0, 1):
        rots = Rotation.from_rotvec(np.zeros((t, 3)))
        out, n = splice_rotation_jumps(rots)
        assert n == 0 and len(out) == t


def test_translation_and_rotation_jumps_are_reported_separately():
    """The report drives the P2 measurement of how much data jumps affect, so the
    two kinds must not be conflated."""
    chunk = make_chunk(t=16, jitter=0.005)
    chunk[5:, :3] += np.array([0.4, 0.0, 0.0])
    extra = Rotation.from_rotvec(np.deg2rad(30) * np.array([0.0, 0.0, 1.0]))
    chunk[11:, 3:6] = (extra * Rotation.from_rotvec(chunk[11:, 3:6])).as_rotvec()
    _rel, rep = to_chunk_relative(chunk)
    assert rep["n_jumps_spliced"] == 1
    assert rep["n_rot_jumps_spliced"] == 1
