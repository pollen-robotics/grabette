"""An uncalibrated grabette must refuse to record, and be fixable from the dashboard.

Without a saved zero the angle offsets are 0, so the "angles" recorded are raw
magnet positions: a gripper channel that converts fine and means nothing. The
fix used to be an SSH session and scripts/calibrate_angles.py plus a restart;
it is now the dashboard's "Calibrate my device" button, which goes through
Backend.calibrate_angles and POST /api/calibration.
"""
import asyncio
import json
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from grabette.hardware import angle as angle_mod


@pytest.fixture
def calib_file(monkeypatch, tmp_path):
    path = tmp_path / "angle_calibration.json"
    monkeypatch.setattr(angle_mod, "CALIBRATION_FILE", path)
    return path


# --- the calibration file ------------------------------------------------------

def test_no_file_is_no_calibration(calib_file):
    assert angle_mod.load_calibration() is None


@pytest.mark.parametrize("content", [
    "not json",
    json.dumps({"sensor_1_offset_deg": 12.0}),  # one sensor only
    json.dumps({"sensor_1_offset_deg": "x", "sensor_2_offset_deg": 3.0}),
])
def test_an_unusable_file_is_no_calibration(calib_file, content):
    calib_file.write_text(content)
    assert angle_mod.load_calibration() is None


def test_saved_calibration_reads_back(calib_file):
    angle_mod.save_calibration(12.3456789, 300.0)

    data = angle_mod.load_calibration()
    assert data["sensor_1_offset_deg"] == pytest.approx(12.345679)
    assert data["sensor_2_offset_deg"] == 300.0
    assert "timestamp" in data


def test_the_average_survives_the_wrap(monkeypatch):
    # A magnet sitting on 0°/360° must average to ~0, not 180.
    monkeypatch.setattr(angle_mod.time, "sleep", lambda s: None)
    reads = iter([359.0, 1.0] * 10)
    avg = angle_mod.read_raw_averaged(lambda: next(reads), n=20)
    assert min(avg, 360.0 - avg) < 1e-6


def test_angle_capture_calibrates_and_applies_at_once(calib_file, monkeypatch):
    monkeypatch.setattr(angle_mod.time, "sleep", lambda s: None)

    def _bus(raw):  # an AS5600 frozen at `raw` (12-bit)
        def rd(addr, out, result):
            result[0], result[1] = raw >> 8, raw & 0xFF
        return types.SimpleNamespace(writeto_then_readfrom=rd)

    cap = angle_mod.AngleCapture(sync_manager=None)
    cap._i2c_1, cap._i2c_2 = _bus(1024), _bus(2048)  # 90°, 180°

    cap.calibrate()

    assert cap._offset_1_deg == pytest.approx(90.0)
    assert cap._offset_2_deg == pytest.approx(180.0)
    assert angle_mod.load_calibration()["sensor_2_offset_deg"] == pytest.approx(180.0)


# --- the RPi backend: fault, gate, fix ----------------------------------------

class _FakeAngle:
    sample_count = 0

    def __init__(self, sync=None):
        self.calibrated = False

    def init_sensors(self):
        pass

    def calibrate(self):
        self.calibrated = True
        return angle_mod.save_calibration(10.0, 20.0)


def _backend(monkeypatch, enable_angle=True):
    from grabette.backend import rpi
    monkeypatch.setattr(angle_mod, "AngleCapture", _FakeAngle)
    b = rpi.RpiBackend(enable_angle=enable_angle, enable_oakd=False)
    if enable_angle:  # what start() does
        b._init_angle_sensors()
        b._check_angle_calibration()
    return b


def test_missing_calibration_blocks_recording(calib_file, monkeypatch):
    b = _backend(monkeypatch)

    assert b.needs_calibration
    assert "not calibrated" in b.hardware_error
    with pytest.raises(RuntimeError, match="not calibrated"):
        b.raise_if_capture_blocked()


def test_a_calibrated_device_is_not_blocked(calib_file, monkeypatch):
    angle_mod.save_calibration(1.0, 2.0)
    b = _backend(monkeypatch)

    assert not b.needs_calibration
    assert b.hardware_error == ""


def test_calibrating_clears_the_fault(calib_file, monkeypatch):
    b = _backend(monkeypatch)

    result = asyncio.run(b.calibrate_angles())

    assert b._angle.calibrated
    assert result["sensor_1_offset_deg"] == 10.0
    assert not b.needs_calibration
    assert b.hardware_error == ""


def test_no_calibration_during_a_recording(calib_file, monkeypatch):
    b = _backend(monkeypatch)
    b._capturing = True

    with pytest.raises(RuntimeError, match="recording"):
        asyncio.run(b.calibrate_angles())
    assert not b._angle.calibrated


def test_an_angle_less_setup_needs_no_calibration(calib_file, monkeypatch):
    b = _backend(monkeypatch, enable_angle=False)

    assert not b.needs_calibration
    with pytest.raises(RuntimeError):
        asyncio.run(b.calibrate_angles())


def test_capture_status_carries_the_flag(calib_file, monkeypatch):
    b = _backend(monkeypatch)

    status = b.get_capture_status()

    assert status.needs_calibration
    assert "not calibrated" in status.blocked_reason


# --- the route the dashboard calls --------------------------------------------

def _client(backend):
    from grabette.app.dependencies import get_backend
    from grabette.app.routers.calibration import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_backend] = lambda: backend
    return TestClient(app)


def test_route_reports_and_calibrates(calib_file, monkeypatch):
    client = _client(_backend(monkeypatch))

    assert client.get("/api/calibration").json() == {"needs_calibration": True}
    r = client.post("/api/calibration")
    assert r.status_code == 200
    assert r.json()["needs_calibration"] is False
    assert client.get("/api/calibration").json() == {"needs_calibration": False}


def test_route_refusal_is_a_conflict_with_the_reason(calib_file, monkeypatch):
    b = _backend(monkeypatch)
    b._capturing = True

    r = _client(b).post("/api/calibration")

    assert r.status_code == 409
    assert "recording" in r.json()["detail"]
