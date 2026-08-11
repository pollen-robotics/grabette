"""Tests for the raw-recording content gate (checks.recording).

Targets the JSON/logic branches (imu, calib, gripper-motion) plus the
end-to-end missing-file aggregation — no real video decode needed (missing
files short-circuit before av.open).
"""
import json

from grabette_postprocess.checks.recording import (
    _check_calib,
    _check_imu,
    check_recording,
    static_gripper_joints,
)


def _write(path, obj):
    path.write_text(json.dumps(obj))


def _status():
    return {"errors": [], "warnings": [], "info": []}


# ── static_gripper_joints (feeds the fixed_gripper tag) ──────────────────

def test_static_gripper_joints_one_static(tmp_path):
    # value = [distal, proximal]: distal moves, proximal static.
    _write(tmp_path / "angle_data.json", {"samples": [
        {"cts": 0, "value": [0.0, 0.2]},
        {"cts": 100, "value": [0.5, 0.2]},
    ]})
    assert static_gripper_joints(tmp_path) == ["proximal"]


def test_static_gripper_joints_both_move(tmp_path):
    _write(tmp_path / "angle_data.json", {"samples": [
        {"cts": 0, "value": [0.0, 0.0]},
        {"cts": 100, "value": [0.5, 0.5]},
    ]})
    assert static_gripper_joints(tmp_path) == []


def test_static_gripper_joints_missing_file(tmp_path):
    assert static_gripper_joints(tmp_path) == []


# ── _check_imu ───────────────────────────────────────────────────────────

def test_check_imu_missing_gyro_errors(tmp_path):
    _write(tmp_path / "oakd_imu.json", {"samples": [
        {"kind": "accel"}, {"kind": "accel"}, {"kind": "rotation"},
    ]})
    st = _status()
    _check_imu(tmp_path, st)
    assert any("no gyro" in e for e in st["errors"])


def test_check_imu_ok_but_no_rotation_warns(tmp_path):
    _write(tmp_path / "oakd_imu.json", {"samples": [
        {"kind": "accel"}, {"kind": "gyro"},
    ]})
    st = _status()
    _check_imu(tmp_path, st)
    assert not st["errors"]
    assert any("rotation" in w for w in st["warnings"])


def test_check_imu_missing_file_is_not_an_error(tmp_path):
    # The Orbbec Gemini 305 has no IMU, so an episode from it carries no
    # oakd_imu.json. That must not be flagged: convert.py and offline_vslam both
    # handle the absence, and Phase 0 measured IMU-free odometry as
    # indistinguishable from re-running the pipeline. Reporting it as an error
    # would mark every valid 305 recording as broken.
    st = _status()
    _check_imu(tmp_path, st)
    assert not st["errors"]
    assert any("IMU-less" in i for i in st["info"])


def test_check_imu_present_but_empty_still_errors(tmp_path):
    # An OAK-D that wrote the file but captured nothing is still a real failure;
    # relaxing the missing-file case must not relax this one.
    _write(tmp_path / "oakd_imu.json", {"samples": []})
    st = _status()
    _check_imu(tmp_path, st)
    assert any("accel" in e for e in st["errors"])
    assert any("gyro" in e for e in st["errors"])


# ── _check_calib ─────────────────────────────────────────────────────────

def test_check_calib_bad_intrinsics(tmp_path):
    _write(tmp_path / "oakd_calib_offline.json", {
        "width": 640, "height": 400, "fx": 0.0, "fy": 100.0,
        "cx": 320, "cy": 200, "baseline": 0.05,
    })
    st = _status()
    _check_calib(tmp_path, st)
    assert any("intrinsics" in e for e in st["errors"])


def test_check_calib_missing_key(tmp_path):
    _write(tmp_path / "oakd_calib_offline.json", {
        "width": 640, "height": 400, "fx": 100, "fy": 100, "cx": 320, "cy": 200,
        # baseline + imu_to_cam missing
    })
    st = _status()
    _check_calib(tmp_path, st)
    assert any("missing keys" in e for e in st["errors"])


def test_check_calib_without_imu_to_cam_is_valid(tmp_path):
    # An IMU-less camera writes no imu_to_cam. offline_vslam probes for it with
    # contains() and falls back to identity, so requiring it here would reject
    # every Gemini 305 episode.
    _write(tmp_path / "oakd_calib_offline.json", {
        "width": 640, "height": 400, "fx": 311.4, "fy": 311.4,
        "cx": 318.75, "cy": 196.25, "baseline": 0.018156,
    })
    st = _status()
    _check_calib(tmp_path, st)
    assert not st["errors"]


# ── check_recording aggregate (empty dir → all required inputs missing) ──

def test_check_recording_empty_dir_reports_missing(tmp_path):
    status = check_recording(tmp_path)
    errs = " ".join(status["errors"])
    for expected in ("oakd_left.mp4", "angle_data.json",
                     "oakd_calib_offline.json"):
        assert expected in errs
    # oakd_imu.json is deliberately NOT required — see
    # test_check_imu_missing_file_is_not_an_error.
    assert "oakd_imu.json" not in errs
