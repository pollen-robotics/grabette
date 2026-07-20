"""End-to-end tests for the raw-recording quality gate (checks.recording).

check_recording is what stands between a raw capture and the SLAM/dataset
pipeline. These tests build a complete valid episode, assert it passes cleanly,
then inject one defect at a time and assert the specific failure is caught — the
behaviour the pipeline actually relies on.
"""

import json

from grabette_postprocess.checks import recording
from grabette_postprocess.checks.recording import (
    _check_calib,
    _check_depth,
    _check_imu,
    _check_seq_overlap,
    check_recording,
    static_gripper_joints,
)


def _rewrite(path, mutate):
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))


def _status():
    return {"name": "ep", "errors": [], "warnings": [], "info": []}


# ---- the happy path -------------------------------------------------------

def test_valid_episode_passes_cleanly(valid_episode):
    """A complete, well-formed episode reports zero errors and zero warnings."""
    status = check_recording(valid_episode, require_right=True)
    assert status["errors"] == []
    assert status["warnings"] == []
    # The info line records what it saw — the slam frame estimate must be > 0.
    assert any(i.startswith("slam_frames~") and not i.endswith("~0")
               for i in status["info"])


# ---- missing inputs are hard errors --------------------------------------

def test_empty_directory_reports_every_required_input(tmp_path):
    """An empty directory flags every required input as an error."""
    errors = " ".join(check_recording(tmp_path, require_right=True)["errors"])
    for needle in ("raw_video.mp4", "oakd_left.mp4", "oakd_left_timestamps.json",
                   "depth", "oakd_imu.json", "angle_data.json",
                   "oakd_calib_offline.json"):
        assert needle in errors, f"missing-input check did not flag {needle}"


def test_empty_video_is_an_error(valid_episode):
    """A present-but-zero-byte video is reported as empty."""
    (valid_episode / "raw_video.mp4").write_bytes(b"")
    errors = " ".join(check_recording(valid_episode)["errors"])
    assert "raw_video.mp4" in errors and "empty" in errors


# ---- SLAM frame count: left ∩ depth seq overlap --------------------------

def test_seq_overlap_present_yields_frames(tmp_path):
    """Overlapping left/depth seqs report a positive SLAM frame estimate, no error."""
    samples = [{"seq": i} for i in range(5)]
    (tmp_path / "oakd_left_timestamps.json").write_text(json.dumps({"samples": samples}))
    (tmp_path / "oakd_depth_timestamps.json").write_text(json.dumps({"samples": samples}))
    status = _status()
    _check_seq_overlap(tmp_path, status)
    assert status["errors"] == []
    assert "slam_frames~5" in status["info"]


def test_disjoint_seqs_would_give_slam_zero_frames(tmp_path):
    """Non-overlapping left/depth seqs are an error (SLAM would get 0 frames)."""
    (tmp_path / "oakd_left_timestamps.json").write_text(
        json.dumps({"samples": [{"seq": i} for i in range(5)]}))
    (tmp_path / "oakd_depth_timestamps.json").write_text(
        json.dumps({"samples": [{"seq": i} for i in range(100, 105)]}))
    status = _status()
    _check_seq_overlap(tmp_path, status)
    assert any("share no seq" in e for e in status["errors"])


# ---- IMU: accel+gyro required, rotation optional -------------------------

def test_imu_missing_gyro_is_error(valid_episode):
    """Missing gyro samples are a hard error (required SLAM input)."""
    _rewrite(valid_episode / "oakd_imu.json",
             lambda d: d.__setitem__(
                 "samples", [s for s in d["samples"] if s["kind"] != "gyro"]))
    status = _status()
    _check_imu(valid_episode, status)
    assert any("no gyro" in e for e in status["errors"])


def test_imu_missing_rotation_only_warns(valid_episode):
    """Missing rotation samples only warn (optional VIO init aid)."""
    _rewrite(valid_episode / "oakd_imu.json",
             lambda d: d.__setitem__(
                 "samples", [s for s in d["samples"] if s["kind"] != "rotation"]))
    status = _status()
    _check_imu(valid_episode, status)
    assert status["errors"] == []
    assert any("rotation" in w for w in status["warnings"])


# ---- calibration: required keys + positive intrinsics --------------------

def test_calib_missing_key_is_error(valid_episode):
    """A calibration file missing a required key is an error naming that key."""
    _rewrite(valid_episode / "oakd_calib_offline.json",
             lambda d: d.pop("baseline"))
    status = _status()
    _check_calib(valid_episode, status)
    assert any("missing keys" in e and "baseline" in e for e in status["errors"])


def test_calib_nonpositive_intrinsics_is_error(valid_episode):
    """Non-positive intrinsics (fx=0) are flagged as bad intrinsics."""
    _rewrite(valid_episode / "oakd_calib_offline.json",
             lambda d: d.__setitem__("fx", 0.0))
    status = _status()
    _check_calib(valid_episode, status)
    assert any("bad intrinsics" in e for e in status["errors"])


# ---- gripper motion -------------------------------------------------------

def test_static_gripper_is_flagged(valid_episode):
    """A gripper whose joints never move is reported static (both joints) and warns."""
    # Freeze both joints -> both reported static, and check_recording warns.
    frozen = {"samples": [{"cts": i * 33, "value": [1.0, 2.0]} for i in range(30)]}
    (valid_episode / "angle_data.json").write_text(json.dumps(frozen))
    assert set(static_gripper_joints(valid_episode)) == {"distal", "proximal"}
    warnings = " ".join(check_recording(valid_episode)["warnings"])
    assert "static" in warnings


def test_moving_gripper_is_not_flagged(valid_episode):
    """A gripper that moves (the valid fixture) reports no static joints."""
    assert static_gripper_joints(valid_episode) == []


# ---- depth stream ---------------------------------------------------------

def test_missing_depth_is_error(valid_episode):
    """Removing the depth stream (no mkv, no png dir) is a hard error."""
    (valid_episode / "oakd_depth.mkv").unlink()
    status = _status()
    _check_depth(valid_episode, status)
    assert any("missing depth" in e for e in status["errors"])
