"""Tests for the SLAM trajectory quality judge (checks.trajectory.check_trajectory).

Each test writes a synthetic camera_trajectory.csv crafted to hit one
verdict/branch. check_trajectory only reads is_lost/x/y/z/timestamp; the other
columns are filled with the real schema for realism.
"""
import numpy as np
import pandas as pd

from grabette_postprocess.checks.trajectory import check_trajectory

COLS = ["frame_idx", "timestamp", "state", "is_lost", "is_keyframe",
        "x", "y", "z", "q_x", "q_y", "q_z", "q_w"]


def _write_traj(path, xyz, timestamps, is_lost=None):
    xyz = np.asarray(xyz, dtype=float)
    n = len(xyz)
    is_lost = np.zeros(n, dtype=int) if is_lost is None else np.asarray(is_lost)
    rows = [[i, float(timestamps[i]), "OK", int(is_lost[i]), 0,
             xyz[i, 0], xyz[i, 1], xyz[i, 2], 0, 0, 0, 1] for i in range(n)]
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)
    return path


def test_good_trajectory(tmp_path):
    xyz = np.zeros((10, 3))
    xyz[:, 0] = np.arange(10) * 0.005            # 5mm straight steps
    xyz[:, 1] = (np.arange(10) % 2) * 0.001       # tiny wiggle -> varied angle
    t = np.linspace(0, 0.9, 10)                   # low speed
    r = check_trajectory(_write_traj(tmp_path / "camera_trajectory.csv", xyz, t))
    assert r.verdict == "GOOD"
    assert not r.errors and not r.warnings
    assert r.n_frames == 10 and r.n_lost == 0 and r.tracking_pct == 100.0
    assert r.n_jumps == 0


def test_fail_too_few_tracked(tmp_path):
    xyz = np.zeros((3, 3))
    r = check_trajectory(_write_traj(
        tmp_path / "camera_trajectory.csv", xyz, [0, 0.1, 0.2], is_lost=[1, 1, 1]))
    assert r.verdict == "FAIL"
    assert any("tracked" in e for e in r.errors)


def test_imu_drift_detected(tmp_path):
    # 20mm straight steps (median>15, angle~0<5), low speed -> IMU drift only.
    xyz = np.zeros((10, 3))
    xyz[:, 0] = np.arange(10) * 0.02
    r = check_trajectory(_write_traj(
        tmp_path / "camera_trajectory.csv", xyz, np.linspace(0, 1.0, 10)))
    assert r.verdict == "BAD"
    assert any("drift" in e.lower() for e in r.errors)
    assert not any("speed" in e.lower() for e in r.errors)


def test_zigzag_detected(tmp_path):
    # 100mm jumps reversing direction (angle~180) -> zigzag.
    xyz = np.zeros((8, 3))
    xyz[:, 0] = np.array([0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1])
    r = check_trajectory(_write_traj(
        tmp_path / "camera_trajectory.csv", xyz, np.linspace(0, 1.0, 8)))
    assert r.verdict == "BAD"
    assert any("zigzag" in e.lower() for e in r.errors)


def test_unrealistic_speed_flagged(tmp_path):
    # Fast motion (0.9m in 0.2s) -> speed error present (drift may also fire).
    xyz = np.zeros((10, 3))
    xyz[:, 0] = np.arange(10) * 0.1
    r = check_trajectory(_write_traj(
        tmp_path / "camera_trajectory.csv", xyz, np.linspace(0, 0.2, 10)))
    assert r.verdict == "BAD"
    assert any("speed" in e.lower() for e in r.errors)


def test_low_tracking_warns(tmp_path):
    # 4/10 tracked (40% < 50), tiny steps -> WARN, no errors.
    xyz = np.zeros((10, 3))
    xyz[:, 0] = np.arange(10) * 0.003
    is_lost = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    r = check_trajectory(_write_traj(
        tmp_path / "camera_trajectory.csv", xyz, np.linspace(0, 1.0, 10), is_lost))
    assert r.tracking_pct == 40.0
    assert r.verdict == "WARN"
    assert any("tracking" in w.lower() for w in r.warnings)
    assert not r.errors
