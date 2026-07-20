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


def test_check_imu_missing_file_errors(tmp_path):
    st = _status()
    _check_imu(tmp_path, st)
    assert any("missing oakd_imu.json" in e for e in st["errors"])


# ── _check_calib ─────────────────────────────────────────────────────────

def test_check_calib_bad_intrinsics(tmp_path):
    _write(tmp_path / "oakd_calib_offline.json", {
        "width": 640, "height": 400, "fx": 0.0, "fy": 100.0,
        "cx": 320, "cy": 200, "baseline": 0.05, "imu_to_cam": [],
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


# ── check_recording aggregate (empty dir → all required inputs missing) ──

def test_check_recording_empty_dir_reports_missing(tmp_path):
    status = check_recording(tmp_path)
    errs = " ".join(status["errors"])
    for expected in ("oakd_left.mp4", "oakd_imu.json",
                     "angle_data.json", "oakd_calib_offline.json"):
        assert expected in errs
