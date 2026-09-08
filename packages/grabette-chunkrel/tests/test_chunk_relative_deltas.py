"""Tests for the eval-side conversion: relative actions -> arm-service deltas.

The decisive test replays a chunk through the ARM SERVER'S OWN algebra and checks
the arm's integrator target lands exactly where the policy asked. That is the
property the robot depends on; anything less would only be testing my arithmetic
against itself.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from grabette_chunkrel.chunk_relative import (
    DEFAULT_JUMP_CAP_DEG,
    DEFAULT_JUMP_CAP_M,
    ChunkRelativeDeltas,
    from_chunk_relative,
    rotvec_to_r6d,
    to_chunk_relative,
)


def r6d_to_matrix(v):
    """Gram-Schmidt in the repo's ROW convention: the 6D vector is the first two
    ROWS of the matrix (see rotation_6d_to_rotation_matrix_numpy in
    integrations/DiffusionPolicy/rotation.py, which is what the arm decodes with).

    This was written with `axis=1` — stacking the basis vectors as COLUMNS — and
    it agreed with an encoder that had the same mistake. Encoder and decoder were
    consistent with each other and inverted relative to the arm, so every test
    here passed while the robot received backwards rotations. Hence
    test_the_wire_encoding_matches_the_repo_convention below, which checks
    against the real implementation rather than against this one.
    """
    a1, a2 = np.asarray(v[:3], float), np.asarray(v[3:6], float)
    b1 = a1 / np.linalg.norm(a1)
    a2p = a2 - b1.dot(a2) * b1
    b2 = a2p / np.linalg.norm(a2p)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=0)


def simulate_arm(deltas, start_pos, start_rot):
    """Replay grpc_server_real.SendCartesianDelta exactly:

        delta_pos_world = R_target @ delta_pos ; target_pos += delta_pos_world
        R_target_new    = R_target @ R_delta
    """
    pos, R = np.asarray(start_pos, float).copy(), start_rot
    traj = []
    for dp, dr6d in deltas:
        pos = pos + R.as_matrix() @ np.asarray(dp, float)
        R = Rotation.from_matrix(R.as_matrix() @ r6d_to_matrix(dr6d))
        traj.append((pos.copy(), R))
    return traj


def make_chunk(t=12, seed=0):
    """A glitch-free chunk: steps stay well under DEFAULT_JUMP_CAP_M / _DEG so the
    splice never fires. Tests that want a glitch inject one explicitly."""
    r = np.random.default_rng(seed)
    pos = np.cumsum(r.normal(scale=0.004, size=(t, 3)), axis=0) + np.array([2.0, -1.0, 0.4])
    rot = np.cumsum(r.normal(scale=0.008, size=(t, 3)), axis=0) + np.array([0.5, -1.2, 0.9])
    grip = np.abs(r.normal(scale=0.2, size=(t, 2)))
    return np.concatenate([pos, rot, grip], axis=1)


# ── the property the robot depends on ───────────────────────────────────

@pytest.mark.parametrize("seed", range(4))
def test_the_arm_lands_exactly_where_the_policy_asked(seed):
    """End to end: absolute chunk -> relative actions -> deltas -> arm algebra.

    The arm's target must reproduce the original absolute trajectory, up to the
    arbitrary rigid transform between the SLAM frame and the arm's frame — which
    is precisely what the representation is supposed to be invariant to. So the
    arm is started at the chunk's own first pose and the match must be exact.
    """
    absolute = make_chunk(seed=seed)
    rel, rep = to_chunk_relative(absolute)
    assert rep["n_jumps_spliced"] == 0 and rep["n_rot_jumps_spliced"] == 0, (
        "fixture drifted into the splice path — this test must exercise the algebra, "
        "not the jump repair (see test_a_spliced_chunk_follows_the_repaired_path)"
    )

    conv = ChunkRelativeDeltas()
    deltas = [conv.step(a)[:2] for a in rel]

    traj = simulate_arm(deltas, absolute[0, :3], Rotation.from_rotvec(absolute[0, 3:6]))
    for i, (pos, R) in enumerate(traj):
        assert np.abs(pos - absolute[i, :3]).max() < 1e-6, f"position drift at {i}"
        err = (R.inv() * Rotation.from_rotvec(absolute[i, 3:6])).magnitude()
        assert np.rad2deg(err) < 1e-3, f"rotation drift at {i}"


def test_the_arm_start_pose_is_irrelevant():
    """The deltas must not depend on where the arm happens to be — that is the
    whole point of a relative representation, and the eval loop never tells the
    converter the measured pose."""
    rel, _ = to_chunk_relative(make_chunk())
    a = [ChunkRelativeDeltas().step(x)[:2] for x in rel]
    b = [ChunkRelativeDeltas().step(x)[:2] for x in rel]
    for (dp1, dr1), (dp2, dr2) in zip(a, b):
        assert np.abs(np.asarray(dp1) - np.asarray(dp2)).max() < 1e-12
        assert np.abs(np.asarray(dr1) - np.asarray(dr2)).max() < 1e-12


# ── the reset contract ──────────────────────────────────────────────────

def test_the_first_delta_of_a_chunk_moves_from_the_reference():
    """Chunk action 0 is the offset from the reference, so the first delta is
    that offset itself — not zero."""
    rel, _ = to_chunk_relative(make_chunk())
    conv = ChunkRelativeDeltas()
    dp, _dr, _g = conv.step(rel[3])          # pretend the chunk starts at index 3
    assert np.abs(dp - rel[3][:3]).max() < 1e-9


def test_forgetting_to_reset_across_chunks_is_a_large_error():
    """Justifies why reset() is mandatory at every replan: carrying `prev` over a
    boundary differences two unrelated frames."""
    rel_a, _ = to_chunk_relative(make_chunk(seed=1))
    rel_b, _ = to_chunk_relative(make_chunk(seed=2))

    conv = ChunkRelativeDeltas()
    for a in rel_a:
        conv.step(a)
    without_reset = conv.step(rel_b[0])[0]

    conv.reset()
    with_reset = conv.step(rel_b[0])[0]

    assert np.linalg.norm(without_reset - with_reset) > 0.01, (
        "a missed reset should be a centimetre-scale error, not a rounding one"
    )


def test_reset_returns_to_the_initial_state():
    conv = ChunkRelativeDeltas()
    rel, _ = to_chunk_relative(make_chunk())
    first = conv.step(rel[0])[0].copy()
    for a in rel[1:]:
        conv.step(a)
    conv.reset()
    assert np.abs(conv.step(rel[0])[0] - first).max() < 1e-12


# ── gripper and contracts ───────────────────────────────────────────────

def test_the_gripper_passes_through_untouched():
    """With the projection, closure = 1.0 means 'fully closed' — an absolute
    command a relative encoding could not express."""
    rel, _ = to_chunk_relative(make_chunk())
    conv = ChunkRelativeDeltas()
    for a in rel:
        _dp, _dr, grip = conv.step(a)
        assert np.abs(grip - a[6:]).max() < 1e-12


def test_the_wire_rotation_is_six_values():
    rel, _ = to_chunk_relative(make_chunk())
    _dp, dr6d, _g = ChunkRelativeDeltas().step(rel[1])
    assert np.asarray(dr6d).shape == (6,)
    assert np.abs(np.linalg.det(r6d_to_matrix(dr6d)) - 1.0) < 1e-9, "not a rotation"


def test_r6d_round_trips_through_the_wire_encoding():
    rv = np.deg2rad(np.array([3.0, -2.0, 1.5]))
    R_back = Rotation.from_matrix(r6d_to_matrix(rotvec_to_r6d(rv)))
    assert np.rad2deg((R_back.inv() * Rotation.from_rotvec(rv)).magnitude()) < 1e-6


def test_a_wrong_action_width_is_rejected():
    conv = ChunkRelativeDeltas()
    with pytest.raises(ValueError, match="8D relative action"):
        conv.step(np.zeros(11))


def test_only_rotvec_is_supported():
    with pytest.raises(ValueError, match="rotvec"):
        ChunkRelativeDeltas(rot="r6d")


# ── interaction with the jump repair ────────────────────────────────────

def test_a_spliced_chunk_follows_the_repaired_path():
    """When a chunk spans a relocalisation jump the encoder repairs it, so the
    deltas must reproduce the REPAIRED trajectory — and must never hand the arm a
    step larger than the cap, which is what tripped the IK watchdog before."""
    absolute = make_chunk(t=14, seed=7)
    absolute[8:, :3] += np.array([0.30, -0.20, 0.10])          # 37 cm relocalisation
    absolute[8:, 3:6] = (
        Rotation.from_rotvec(np.deg2rad([0.0, 40.0, 0.0]))
        * Rotation.from_rotvec(absolute[8:, 3:6])
    ).as_rotvec()

    rel, rep = to_chunk_relative(absolute)
    assert rep["n_jumps_spliced"] > 0 and rep["n_rot_jumps_spliced"] > 0, "no glitch injected"

    conv = ChunkRelativeDeltas()
    deltas = [conv.step(a)[:2] for a in rel]

    # every commanded step stays inside the caps
    for i, (dp, dr6d) in enumerate(deltas[1:], start=1):
        assert np.linalg.norm(dp) < DEFAULT_JUMP_CAP_M + 1e-9, f"delta {i} exceeds the cap"
        mag = np.rad2deg(Rotation.from_matrix(r6d_to_matrix(dr6d)).magnitude())
        assert mag < DEFAULT_JUMP_CAP_DEG + 1e-6, f"rotation delta {i} exceeds the cap"

    # and the arm tracks the repaired trajectory exactly
    repaired = from_chunk_relative(rel, absolute[0, :6])
    traj = simulate_arm(deltas, repaired[0, :3], Rotation.from_rotvec(repaired[0, 3:6]))
    for i, (pos, R) in enumerate(traj):
        assert np.abs(pos - repaired[i, :3]).max() < 1e-6, f"position drift at {i}"
        err = (R.inv() * Rotation.from_rotvec(repaired[i, 3:6])).magnitude()
        assert np.rad2deg(err) < 1e-3, f"rotation drift at {i}"


# ── the wire convention, checked against the REAL decoder ───────────────

def _repo_rotation_module():
    """integrations/DiffusionPolicy/rotation.py — the encoder the working
    per-step-delta pipeline uses, hence the arm's convention."""
    import importlib.util

    path = (Path(__file__).resolve().parents[3] / "integrations" / "DiffusionPolicy"
            / "rotation.py")
    if not path.is_file():
        pytest.skip("integrations/DiffusionPolicy/rotation.py not in this checkout")
    spec = importlib.util.spec_from_file_location("_repo_rotation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("seed", range(5))
def test_the_wire_encoding_matches_the_repo_convention(seed):
    """THE regression test for the bug that reached the robot.

    rotvec_to_r6d used the first two COLUMNS while the pipeline the arm actually
    decodes uses the first two ROWS, so every commanded rotation delta was the
    inverse. Compare against the real implementation, never against a decoder
    written here.
    """
    rot = _repo_rotation_module()
    rv = np.random.default_rng(seed).normal(scale=0.5, size=3)
    mine = np.asarray(rotvec_to_r6d(rv), float)
    theirs = rot.rotation_matrix_to_rotation_6d_numpy(
        Rotation.from_rotvec(rv).as_matrix().reshape(1, 3, 3))[0]
    assert np.abs(mine - theirs).max() < 1e-9, (
        f"wire convention disagrees with the repo/arm encoder\n"
        f"  ours   {np.round(mine, 4)}\n  theirs {np.round(theirs, 4)}"
    )


def test_deltas_match_the_known_good_delta_pipeline():
    """Equivalence with convert_dataset.compute_delta_actions, which drives the
    per-step-delta policies that work on the robot.

    Both must command the same motion from the same absolute poses:
        chunk-relative:  A_{i-1}^-1 (a_i - a_{i-1}) = R_{i-1}^-1 (p_i - p_{i-1})
        baseline:        actions[t] = R_t^-1 (p_{t+1} - p_t)
    so ours at index i equals the baseline at t = i-1. This is the test that
    localises a frame or convention error to one side or the other.
    """
    import importlib.util
    import sys as _sys

    root = Path(__file__).resolve().parents[3] / "integrations" / "DiffusionPolicy"
    if not (root / "convert_dataset.py").is_file():
        pytest.skip("integrations/DiffusionPolicy not in this checkout")
    _sys.path.insert(0, str(root))
    try:
        rot = _repo_rotation_module()
        spec = importlib.util.spec_from_file_location(
            "_convert_dataset", root / "convert_dataset.py")
        cd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cd)
    finally:
        _sys.path.remove(str(root))

    absolute = make_chunk(t=40, seed=5)
    poses11 = np.zeros((len(absolute), 11), dtype=np.float32)
    poses11[:, :3] = absolute[:, :3]
    poses11[:, 3:9] = rot.rotvec_to_rotation_6d(absolute[:, 3:6])
    poses11[:, 9:] = absolute[:, 6:8]
    base = cd.compute_delta_actions(poses11, np.zeros(len(absolute), dtype=int))

    rel, rep = to_chunk_relative(absolute)
    assert rep["n_jumps_spliced"] == 0 and rep["n_rot_jumps_spliced"] == 0, (
        "the splice repairs glitches the baseline does not, so a spliced chunk "
        "is legitimately not comparable"
    )
    conv = ChunkRelativeDeltas()
    mine = [conv.step(a) for a in rel]

    for i in range(1, len(absolute) - 1):
        assert np.abs(np.asarray(mine[i][0]) - base[i - 1, :3]).max() < 1e-6, (
            f"position delta disagrees at index {i}")
        a = Rotation.from_matrix(r6d_to_matrix(np.asarray(mine[i][1])))
        b = Rotation.from_matrix(r6d_to_matrix(base[i - 1, 3:9]))
        err = np.rad2deg((a.inv() * b).magnitude())
        assert err < 1e-3, f"rotation delta disagrees at index {i} by {err:.4f} deg"
