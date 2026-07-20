"""Tests for the post-SLAM trajectory quality gate (checks.trajectory).

check_trajectory is the counterpart of check_recording, run *after* SLAM: it
judges whether a reconstructed camera path is trustworthy or shows the classic
SLAM failure modes (too few tracked frames, unrealistic speed, IMU dead-reckoning
drift, relocalization zigzag, low tracking). Each test builds a trajectory CSV
exhibiting one behaviour and asserts the verdict/flag the pipeline relies on.
"""

import json

import pandas as pd

from grabette_postprocess.checks.trajectory import check_trajectory


def _write_traj(path, xs, ys, zs, ts, lost):
    df = pd.DataFrame({
        "frame_idx": range(len(xs)),
        "timestamp": ts,
        "is_lost": [int(b) for b in lost],
        "x": xs, "y": ys, "z": zs,
        "q_x": 0.0, "q_y": 0.0, "q_z": 0.0, "q_w": 1.0,
    })
    df.to_csv(path, index=False)
    return path


def test_healthy_trajectory_is_good(tmp_path):
    """A smooth, slow, fully-tracked path with small steps verdicts GOOD."""
    n = 10
    xs = [i * 0.005 for i in range(n)]   # 5 mm steps (< 15 mm, < 50 mm jump)
    ts = [i * 0.1 for i in range(n)]     # 0.05 m/s -> realistic
    traj = _write_traj(tmp_path / "camera_trajectory.csv",
                       xs, [0.0] * n, [0.0] * n, ts, [False] * n)
    report = check_trajectory(traj)
    assert report.verdict == "GOOD"
    assert report.tracking_pct == 100.0
    assert report.n_tracked == n and report.n_lost == 0
    assert report.n_jumps == 0
    assert report.errors == [] and report.warnings == []


def test_too_few_tracked_frames_fails(tmp_path):
    """Fewer than 2 tracked frames is an immediate FAIL."""
    lost = [False, True, True, True, True]
    traj = _write_traj(tmp_path / "camera_trajectory.csv",
                       [0.0] * 5, [0.0] * 5, [0.0] * 5, [i * 0.1 for i in range(5)], lost)
    report = check_trajectory(traj)
    assert report.verdict == "FAIL"
    assert report.n_tracked == 1
    assert any("tracked frames" in e for e in report.errors)


def test_unrealistic_speed_is_bad(tmp_path):
    """A path covering meters in milliseconds flags an unrealistic average speed."""
    n = 10
    xs = [i * 1.0 for i in range(n)]     # 1 m per 0.1 s -> 10 m/s
    ts = [i * 0.1 for i in range(n)]
    traj = _write_traj(tmp_path / "camera_trajectory.csv",
                       xs, [0.0] * n, [0.0] * n, ts, [False] * n)
    report = check_trajectory(traj)
    assert report.verdict == "BAD"
    assert any("Unrealistic avg speed" in e for e in report.errors)


def test_imu_drift_is_bad(tmp_path):
    """Big straight-line steps with no direction change read as IMU dead-reckoning drift."""
    n = 10
    xs = [i * 0.02 for i in range(n)]    # 20 mm steps (> 15 mm), perfectly straight
    ts = [i * 0.1 for i in range(n)]     # 0.2 m/s -> not "unrealistic", isolates drift
    traj = _write_traj(tmp_path / "camera_trajectory.csv",
                       xs, [0.0] * n, [0.0] * n, ts, [False] * n)
    report = check_trajectory(traj)
    assert report.verdict == "BAD"
    assert any("IMU drift" in e for e in report.errors)


def test_zigzag_is_bad(tmp_path):
    """Many large back-and-forth jumps (median direction change > 90°) flag a zigzag."""
    n = 8
    xs = [0.1 * (i % 2) for i in range(n)]  # 0,0.1,0,0.1,... -> 100 mm reversing steps
    ts = [i * 0.1 for i in range(n)]
    traj = _write_traj(tmp_path / "camera_trajectory.csv",
                       xs, [0.0] * n, [0.0] * n, ts, [False] * n)
    report = check_trajectory(traj)
    assert report.verdict == "BAD"
    assert any("Zigzag" in e for e in report.errors)


def test_low_tracking_only_warns(tmp_path):
    """A clean but sparsely-tracked path warns (not errors) -> verdict WARN."""
    # 4 tracked (good motion) + 6 lost -> 40% tracking, no errors.
    n = 10
    lost = [False] * 4 + [True] * 6
    xs = [i * 0.005 for i in range(n)]
    ts = [i * 0.1 for i in range(n)]
    traj = _write_traj(tmp_path / "camera_trajectory.csv",
                       xs, [0.0] * n, [0.0] * n, ts, lost)
    report = check_trajectory(traj)
    assert report.verdict == "WARN"
    assert report.tracking_pct == 40.0
    assert report.errors == []
    assert any("Low tracking" in w for w in report.warnings)


def test_metadata_is_read_when_present(tmp_path):
    """method/frame_skip are populated from slam_metadata.json when provided."""
    n = 10
    xs = [i * 0.005 for i in range(n)]
    ts = [i * 0.1 for i in range(n)]
    traj = _write_traj(tmp_path / "camera_trajectory.csv",
                       xs, [0.0] * n, [0.0] * n, ts, [False] * n)
    meta = tmp_path / "slam_metadata.json"
    meta.write_text(json.dumps({"method": "rgbd", "frame_skip": 2}))
    report = check_trajectory(traj, meta_path=meta)
    assert report.method == "rgbd"
    assert report.frame_skip == 2
