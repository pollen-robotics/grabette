"""Unit tests for grabette_postprocess.oak_slam pure logic (no docker / no SLAM)."""

import json

import numpy as np
import pandas as pd

from grabette_postprocess.oak_slam import (
    SlamResult,
    _estimate_gravity_imu,
    _gravity_align_trajectory,
    _integrate_deltas,
)


def _result(total, tracked):
    return SlamResult(returncode=0, total_frames=total, tracked_frames=tracked,
                      trajectory_path=None)


def test_tracking_pct():
    """tracking_pct is tracked/total as a percentage."""
    assert _result(200, 100).tracking_pct == 50.0
    assert _result(4, 4).tracking_pct == 100.0


def test_tracking_pct_zero_frames_is_zero_not_nan():
    """Zero total frames yields 0.0, not a division-by-zero NaN."""
    assert _result(0, 0).tracking_pct == 0.0


def _identity_quat_deltas(dx, dy, dz, lost):
    n = len(dx)
    return pd.DataFrame({
        "timestamp_s": np.arange(n, dtype=float),
        "dx": dx, "dy": dy, "dz": dz,
        "dqx": [0.0] * n, "dqy": [0.0] * n, "dqz": [0.0] * n, "dqw": [1.0] * n,
        "lost": lost,
    })


def test_integrate_deltas_accumulates_translation():
    """Frame-to-frame deltas integrate into a running absolute position, in the standard schema."""
    # Pure translation, no rotation: absolute pos is the running sum of deltas.
    df = _identity_quat_deltas([0.0, 1.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0],
                               lost=[False, False, False])
    out = _integrate_deltas(df)
    np.testing.assert_allclose(out["x"].values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(out["y"].values, [0.0, 0.0, 2.0])
    # Output carries the standard trajectory schema.
    assert list(out["frame_idx"]) == [0, 1, 2]
    np.testing.assert_allclose(out["q_w"].values, [1.0, 1.0, 1.0])


def test_integrate_deltas_holds_pose_on_lost_frames():
    """Lost frames accumulate no motion — the absolute pose is held across the gap."""
    # Lost frames accumulate no motion -> pose is held across the gap.
    df = _identity_quat_deltas([1.0, 5.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                               lost=[False, True, False])
    out = _integrate_deltas(df)
    # frame 0: x=1 ; frame 1 lost: held at 1 ; frame 2: 1 + 1 = 2 (the lost delta dropped)
    np.testing.assert_allclose(out["x"].values, [1.0, 1.0, 2.0])
    assert list(out["is_lost"]) == [False, True, False]


# ---- gravity estimation ---------------------------------------------------

def _accel_df(rows):
    return pd.DataFrame(rows, columns=["ax", "ay", "az"])


def test_estimate_gravity_filters_motion_samples():
    """With ≥100 near-g samples, motion outliers are excluded and the g direction survives."""
    # 150 rest samples (specific force ~ +Z at 9.81) + 30 motion outliers far from g.
    rest = [[0.0, 0.0, 9.81]] * 150
    motion = [[0.0, 0.0, 20.0]] * 30      # |20 - 9.81| >> 0.5 -> filtered out
    g = _estimate_gravity_imu(_accel_df(rest + motion))
    np.testing.assert_allclose(g, [0.0, 0.0, 9.81], atol=1e-6)


def test_estimate_gravity_falls_back_to_full_median():
    """Fewer than 100 near-g samples -> median over ALL samples, not just near-g ones."""
    # None of these are within 0.5 of 9.81, so the near-g filter keeps nothing
    # and the fallback median of all az (7,8,9) = 8 is returned.
    g = _estimate_gravity_imu(_accel_df([[0, 0, 7.0], [0, 0, 8.0], [0, 0, 9.0]]))
    np.testing.assert_allclose(g, [0.0, 0.0, 8.0])


# ---- gravity alignment ----------------------------------------------------

def _traj_df(positions, *, lost=None):
    n = len(positions)
    lost = [False] * n if lost is None else lost
    pos = np.asarray(positions, dtype=float)
    return pd.DataFrame({
        "is_lost": [int(b) for b in lost],
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "q_x": 0.0, "q_y": 0.0, "q_z": 0.0, "q_w": 1.0,  # identity orientation
    })


def _write_gravity_inputs(oak_dir, accel_row, imu_to_cam):
    oak_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([accel_row] * 150, columns=["ax", "ay", "az"]).to_csv(
        oak_dir / "imu_acc.csv", index=False)
    (oak_dir / "calib_offline.json").write_text(
        json.dumps({"imu_to_cam": imu_to_cam}))


def test_gravity_align_rotates_world_to_z_up(tmp_path):
    """Gravity measured along world -Y is rotated to world -Z, carrying positions with it."""
    # Specific force along +Y (imu_to_cam = identity, identity first pose) ->
    # physical gravity in world = -Y. Aligning -Y to -Z is a +90° rotation about
    # X, which sends a point at (0,1,0) to (0,0,1).
    oak = tmp_path / "oak"
    _write_gravity_inputs(oak, [0.0, 9.81, 0.0], np.eye(4).tolist())
    traj = _traj_df([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = _gravity_align_trajectory(traj, oak)
    np.testing.assert_allclose(out[["x", "y", "z"]].iloc[0], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(out[["x", "y", "z"]].iloc[1], [0.0, 0.0, 1.0], atol=1e-6)


def test_gravity_align_skips_when_inputs_missing(tmp_path):
    """Missing imu_acc.csv / calib -> the trajectory is returned unchanged (no crash)."""
    traj = _traj_df([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = _gravity_align_trajectory(traj, tmp_path / "empty")
    np.testing.assert_allclose(out[["x", "y", "z"]], traj[["x", "y", "z"]])


def test_gravity_align_noop_when_all_lost(tmp_path):
    """With no tracked frames there is no valid first pose -> returns input unchanged."""
    traj = _traj_df([[1.0, 2.0, 3.0]], lost=[True])
    out = _gravity_align_trajectory(traj, tmp_path / "unused")
    np.testing.assert_allclose(out[["x", "y", "z"]], traj[["x", "y", "z"]])
