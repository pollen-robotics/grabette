"""CALIB_STATUS / CALIBRATE / ANGLES over BLE, for the first-run setup page.

The BLE service does not touch the sensors: it forwards to the daemon's
/api/calibration and /api/state, the routes behind the dashboard's "Calibrate
my device" and its 3D model. These tests stand a tiny HTTP server in for the
daemon.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from grabette.bluetooth import bluetooth_service as bts


class _FakeDaemon:
    """Answers /api/calibration and /api/state like the daemon's routers do."""

    def __init__(self):
        self.needs_calibration = True
        self.angle = {"timestamp_ms": 0.0, "proximal": -0.5, "distal": 0.125}
        self.post_status = 200
        self.post_detail = None
        self.posts = 0

    def handler(self):
        daemon = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/state":
                    self._reply(200, {"imu": None, "angle": daemon.angle})
                else:
                    self._reply(200, {"needs_calibration": daemon.needs_calibration})

            def do_POST(self):
                daemon.posts += 1
                if daemon.post_status != 200:
                    self._reply(daemon.post_status, {"detail": daemon.post_detail})
                    return
                daemon.needs_calibration = False
                self._reply(200, {"needs_calibration": False, "calibration": {}})

        return Handler


@pytest.fixture
def daemon(monkeypatch):
    fake = _FakeDaemon()
    server = HTTPServer(("127.0.0.1", 0), fake.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(bts, "_DAEMON_URL",
                        f"http://127.0.0.1:{server.server_port}")
    yield fake
    server.shutdown()
    server.server_close()


@pytest.fixture
def service():
    return bts.BluetoothWifiService(device_name="Grabette", pin_code="12345")


def _cmd(service, text):
    return service._handle_command(text.encode())


def test_requires_pin(service, daemon):
    assert _cmd(service, "CALIB_STATUS").startswith("ERROR: Not authenticated")
    assert _cmd(service, "CALIBRATE").startswith("ERROR: Not authenticated")
    assert _cmd(service, "ANGLES").startswith("ERROR: Not authenticated")
    assert daemon.posts == 0


def test_status_then_calibrate(service, daemon):
    assert _cmd(service, "PIN_12345") == "OK: Connected"
    assert _cmd(service, "CALIB_STATUS") == "OK: NEEDS_CALIBRATION"
    assert _cmd(service, "CALIBRATE") == "OK: Calibrated"
    assert daemon.posts == 1
    # The PIN is not consumed: checking again needs no new one.
    assert _cmd(service, "calib_status") == "OK: CALIBRATED"


def test_daemon_refusal_is_relayed(service, daemon):
    daemon.post_status = 409
    daemon.post_detail = "Not during a recording."
    _cmd(service, "PIN_12345")
    assert _cmd(service, "CALIBRATE") == "ERROR: Not during a recording."


def test_daemon_down(service, monkeypatch):
    # Nothing listens on port 9 (discard) on a test machine.
    monkeypatch.setattr(bts, "_DAEMON_URL", "http://127.0.0.1:9")
    _cmd(service, "PIN_12345")
    assert _cmd(service, "CALIBRATE").startswith(
        "ERROR: The Grabette service is not running yet")


def test_angles(service, daemon, monkeypatch):
    monkeypatch.setattr(bts.settings, "hand", "left")
    _cmd(service, "PIN_12345")
    assert _cmd(service, "ANGLES") == "OK: hand=left p=-0.5000 d=0.1250"
    # Polled in a loop on one PIN.
    assert _cmd(service, "angles") == "OK: hand=left p=-0.5000 d=0.1250"


def test_angles_without_reading(service, daemon):
    daemon.angle = None
    _cmd(service, "PIN_12345")
    assert _cmd(service, "ANGLES") == "ERROR: The gripper sensors are not answering"
