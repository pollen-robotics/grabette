"""Gripette status web UI — a tiny, isolated health page.

Answers "is the gripette OK?" at a glance: service state, live camera view,
motor positions, system vitals, journal tail — plus Restart-service and
Shutdown buttons (sudoers-gated, see `make install-web`).

Design constraints (why this looks the way it does):
  - SEPARATE process from the gripette service: a UI bug can never take down
    the gripper, and the page survives to report a dead/wedged service.
  - Strictly PULL-based: zero work when no browser is open. No background
    polling threads, no persistent camera stream.
  - Plain stdlib http.server: no new dependencies on the Pi Zero 2W.
  - Health is measured through the SAME gRPC API the eval client uses
    (Ping / ReadMotors / StreamState), so "the page shows an image" proves
    the real pipeline end-to-end. Camera frames are grabbed one at a time
    (stream → first frame → cancel) and cached for FRAME_CACHE_S so several
    viewers cost at most ~1 capture/s on the service.

Run:  python -m gripette.webui      (or via systemd/gripette-web.service)
"""

import json
import logging
import os
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import grpc

from .proto import gripper_pb2, gripper_pb2_grpc

logger = logging.getLogger(__name__)

WEB_PORT = int(os.environ.get("GRIPPER_WEB_PORT", "8080"))
# The gripette gRPC service on this same device.
GRPC_TARGET = f"127.0.0.1:{os.environ.get('GRIPPER_PORT', '50051')}"
GRPC_TIMEOUT_S = 2.0
# Frame grabs ride the service's capture timeout (5 s) + margin.
FRAME_TIMEOUT_S = 8.0
# One grab serves all viewers + status polls within this window.
FRAME_CACHE_S = 1.0
JOURNAL_LINES = 25
STATIC_DIR = Path(__file__).parent / "static"

# Exact commands granted by the sudoers drop-in (make install-web). The
# pre-check, the sudoers rule, and the dispatch must use these identical
# strings or the NOPASSWD match fails (same pattern as grabette).
RESTART_CMD = ("/usr/bin/systemctl", "restart", "gripette")
POWEROFF_CMD = ("/usr/bin/systemctl", "poweroff")


# ---------------------------------------------------------------------------
# gRPC client (lazy singleton — grpc channels are thread-safe and reconnect
# on their own when the service restarts)
# ---------------------------------------------------------------------------

_stub_lock = threading.Lock()
_stub: gripper_pb2_grpc.GripperServiceStub | None = None


def stub() -> gripper_pb2_grpc.GripperServiceStub:
    global _stub
    with _stub_lock:
        if _stub is None:
            channel = grpc.insecure_channel(GRPC_TARGET)
            _stub = gripper_pb2_grpc.GripperServiceStub(channel)
        return _stub


def _rpc_error_detail(e: grpc.RpcError) -> str:
    detail = e.details() if callable(getattr(e, "details", None)) else str(e)
    code = e.code().name if callable(getattr(e, "code", None)) else "?"
    return f"{code}: {detail}"


# ---------------------------------------------------------------------------
# Camera frame (single grab, shared cache)
# ---------------------------------------------------------------------------

_frame_lock = threading.Lock()
_frame_cache = {"jpeg": None, "error": None, "mono": 0.0}


def _grab_one_frame() -> bytes:
    """Open StreamState, take the first frame, cancel the stream."""
    call = stub().StreamState(gripper_pb2.StreamRequest(), timeout=FRAME_TIMEOUT_S)
    try:
        return next(iter(call)).jpeg_data
    finally:
        call.cancel()


def get_frame() -> dict:
    """Return {"jpeg", "error", "age_s"} — at most one real grab per
    FRAME_CACHE_S, shared by all viewers (lock serializes concurrent calls;
    latecomers get the fresh cache)."""
    with _frame_lock:
        age = time.monotonic() - _frame_cache["mono"]
        if age >= FRAME_CACHE_S or (_frame_cache["jpeg"] is None and _frame_cache["error"] is None):
            try:
                _frame_cache["jpeg"] = _grab_one_frame()
                _frame_cache["error"] = None
            except grpc.RpcError as e:
                _frame_cache["jpeg"] = None
                _frame_cache["error"] = _rpc_error_detail(e)
            except Exception as e:
                _frame_cache["jpeg"] = None
                _frame_cache["error"] = str(e)
            _frame_cache["mono"] = time.monotonic()
            age = 0.0
        return {"jpeg": _frame_cache["jpeg"], "error": _frame_cache["error"],
                "age_s": round(age, 2)}


# ---------------------------------------------------------------------------
# Status sources (each best-effort: a missing tool shows as null, never a 500)
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 2.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def service_status() -> dict:
    out = _run(["systemctl", "show", "gripette", "--property",
                "ActiveState,SubState,NRestarts,ExecMainStartTimestamp"])
    props = dict(line.split("=", 1) for line in out.splitlines() if "=" in line) if out else {}
    return {
        "active_state": props.get("ActiveState"),
        "sub_state": props.get("SubState"),
        # NRestarts counts systemd auto-restarts since boot — with the
        # fail-fast machinery this doubles as a camera-wedge counter.
        "n_restarts": props.get("NRestarts"),
        "since": props.get("ExecMainStartTimestamp"),
    }


def journal_tail() -> list[str]:
    out = _run(["journalctl", "-u", "gripette", "-n", str(JOURNAL_LINES),
                "--no-pager", "--output", "short-iso"])
    return out.splitlines() if out else []


def grpc_status() -> dict:
    try:
        resp = stub().Ping(gripper_pb2.PingRequest(), timeout=GRPC_TIMEOUT_S)
        return {"ok": True, "status": resp.status,
                "uptime_s": round(resp.uptime_seconds, 1), "error": None}
    except grpc.RpcError as e:
        return {"ok": False, "status": None, "uptime_s": None,
                "error": _rpc_error_detail(e)}


def motors_status() -> dict:
    try:
        resp = stub().ReadMotors(gripper_pb2.ReadMotorsRequest(), timeout=GRPC_TIMEOUT_S)
        return {"ok": True, "motor1": round(resp.motor1_position, 4),
                "motor2": round(resp.motor2_position, 4), "error": None}
    except grpc.RpcError as e:
        return {"ok": False, "motor1": None, "motor2": None,
                "error": _rpc_error_detail(e)}


def system_info() -> dict:
    info = {"hostname": socket.gethostname()}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info["ip"] = s.getsockname()[0]
        s.close()
    except Exception:
        info["ip"] = None
    try:
        raw = open("/sys/class/thermal/thermal_zone0/temp").read().strip()
        info["cpu_temp_c"] = round(int(raw) / 1000, 1)
    except Exception:
        info["cpu_temp_c"] = None
    # 0x0 = healthy; nonzero bits = under-voltage/throttling (see RPi docs).
    out = _run(["vcgencmd", "get_throttled"])
    info["throttled"] = out.strip().split("=")[-1] if out else None
    try:
        st = os.statvfs("/")
        info["disk_free_gb"] = round(st.f_frsize * st.f_bavail / 1024**3, 1)
    except Exception:
        info["disk_free_gb"] = None
    # /proc/net/wireless: "wlan0: 0000   52.  -58.  ..." → link quality, dBm
    try:
        info["wifi_signal_dbm"] = None
        for line in open("/proc/net/wireless"):
            if ":" in line:
                fields = line.split()
                info["wifi_signal_dbm"] = int(float(fields[3].rstrip(".")))
                break
    except Exception:
        info["wifi_signal_dbm"] = None
    return info


def full_status() -> dict:
    frame = get_frame()  # shares the cache with /api/frame.jpg polls
    return {
        "service": service_status(),
        "grpc": grpc_status(),
        "motors": motors_status(),
        "camera": {"ok": frame["jpeg"] is not None, "error": frame["error"],
                   "age_s": frame["age_s"]},
        "system": system_info(),
        "journal": journal_tail(),
    }


# ---------------------------------------------------------------------------
# Privileged actions (sudoers-gated, same pattern as grabette's shutdown)
# ---------------------------------------------------------------------------

def _sudo_action(cmd: tuple[str, ...], delay_s: float) -> tuple[int, dict]:
    """Pre-check the NOPASSWD grant, then dispatch detached (the HTTP
    response must escape before systemd tears anything down)."""
    check = subprocess.run(["sudo", "-n", "-l", *cmd],
                           capture_output=True, timeout=5)
    if check.returncode != 0:
        return 403, {"status": "forbidden",
                     "detail": "Not permitted — run 'make install-web' on the device."}
    subprocess.Popen(["sh", "-c", f"sleep {delay_s}; sudo -n {' '.join(cmd)}"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 200, {"status": "dispatched"}


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 1 Hz polling would flood the journal

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/":
                html = (STATIC_DIR / "status.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
            elif path == "/api/status":
                self._send_json(200, full_status())
            elif path == "/api/frame.jpg":
                frame = get_frame()
                if frame["jpeg"] is not None:
                    self._send(200, frame["jpeg"], "image/jpeg")
                else:
                    self._send_json(503, {"error": frame["error"]})
            else:
                self._send_json(404, {"error": "not found"})
        except BrokenPipeError:
            pass  # browser navigated away mid-response
        except Exception as e:
            logger.exception("GET %s failed", path)
            try:
                self._send_json(500, {"error": str(e)})
            except Exception:
                pass

    def do_POST(self):
        try:
            if self.path == "/api/restart-service":
                code, body = _sudo_action(RESTART_CMD, delay_s=0.5)
            elif self.path == "/api/shutdown":
                code, body = _sudo_action(POWEROFF_CMD, delay_s=1.0)
            else:
                code, body = 404, {"error": "not found"}
            self._send_json(code, body)
        except Exception as e:
            logger.exception("POST %s failed", self.path)
            try:
                self._send_json(500, {"error": str(e)})
            except Exception:
                pass


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server = ThreadingHTTPServer(("0.0.0.0", WEB_PORT), Handler)
    logger.info("Gripette status page on http://0.0.0.0:%d (gRPC target %s)",
                WEB_PORT, GRPC_TARGET)
    server.serve_forever()


if __name__ == "__main__":
    main()
