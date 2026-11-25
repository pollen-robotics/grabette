"""A casquette that cannot record must say so to the fleet.

What went wrong: the IMU was missing at boot, so the backend ran camera-only and
refused every capture — but the fleet's synchronized start was acked
"scheduled" first, and the capture only failed at T0, logged on the device and
nowhere else. 0 of 50 takes recorded, no message.

Now:
  * the fault rides the heartbeat (`error`) — the fleet refuses the start;
  * a start command is refused up front when a fault is known;
  * a scheduled start that fails at T0 is reported as the take's issue
    "start_failed" (heartbeat `last_episode` / `last_issues`).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from aiohttp import web

import casquette.fleet.relay_client as relay_mod
from casquette.app.main import _handle_relay_command
from casquette.backend.mock import MockBackend
from casquette.backend.rpi import RpiBackend
from casquette.fleet.capture_scheduler import get_capture_scheduler

FAULT = "IMU not detected at startup: recording is impossible."


def _utc_iso(offset_s: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


async def _wait_scheduler_idle(timeout: float = 3.0) -> None:
    sched = get_capture_scheduler()
    for _ in range(int(timeout / 0.02)):
        if not sched.is_scheduled():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("scheduled start never fired")


# --- the backend names its fault ----------------------------------------------

def test_rpi_backend_without_imu_reports_a_hardware_error():
    b = RpiBackend()
    b._running, b._imu, b._imu_init_error = True, None, "[Errno 5] Input/output error"
    assert "IMU" in b.hardware_error and "[Errno 5]" in b.hardware_error


def test_rpi_backend_names_the_imus_it_tried_when_none_is_detected(monkeypatch):
    import casquette.hardware.imu as imu_mod
    monkeypatch.setattr(imu_mod, "detect_imu_capture", lambda *a: None)
    b = RpiBackend(imu_candidates=[("lsm6dsv16x", 1, {}), ("bmi088", 1, {})])
    b._running, b._imu = True, b._detect_imu()
    assert "lsm6dsv16x on bus 1" in b.hardware_error and "bmi088 on bus 1" in b.hardware_error

    monkeypatch.setattr(imu_mod, "detect_imu_capture", lambda *a: object())
    b._imu = b._detect_imu()  # e.g. the re-detection after a stop finds it again
    assert b.hardware_error == ""


def test_rpi_backend_with_imu_or_not_running_is_healthy():
    b = RpiBackend()
    assert b.hardware_error == ""  # not started yet
    b._running, b._imu = True, object()
    assert b.hardware_error == ""


def test_mock_backend_is_healthy():
    assert MockBackend().hardware_error == ""


# --- the start command refuses a known fault up front -------------------------

async def test_start_is_refused_with_the_fault_instead_of_acked(daemon, tm, monkeypatch):
    monkeypatch.setattr(MockBackend, "hardware_error", property(lambda self: FAULT))

    r = await _handle_relay_command({"type": "start_capture", "id": "c1",
                                     "args": {"task_name": "pick", "start_at_utc": _utc_iso(0.2)}})

    assert r == {"status": "error", "message": FAULT}
    assert not get_capture_scheduler().is_scheduled()
    assert tm.pending_episode_id is None


# --- a start that fails at T0 is reported as the take's issue -----------------

async def test_a_scheduled_start_failing_at_t0_reports_start_failed(daemon, tm, monkeypatch):
    async def boom(self, episode_dir):
        raise RuntimeError("Cannot capture: IMU is not available")
    monkeypatch.setattr(MockBackend, "start_capture", boom)

    r = await _handle_relay_command({"type": "start_capture", "id": "c1",
                                     "args": {"task_name": "pick", "start_at_utc": _utc_iso(0.1)}})
    assert r["status"] == "scheduled"  # acked: the fleet believes it records
    await _wait_scheduler_idle()

    assert tm.last_episode_issues() == (r["episode_id"], ["start_failed"])
    assert not tm.episode_dir(r["episode_id"]).exists()  # nothing half-made left behind


async def test_a_start_cancelled_before_t0_is_not_a_failure(daemon, tm):
    r = await _handle_relay_command({"type": "start_capture", "id": "c1",
                                     "args": {"task_name": "pick", "start_at_utc": _utc_iso(5)}})
    assert r["status"] == "scheduled"
    r2 = await _handle_relay_command({"type": "stop_capture", "id": "c2", "args": {}})
    assert r2["status"] == "cancelled"

    assert tm.last_episode_issues() is None


# --- a saved take reports what is wrong with it --------------------------------

def _saved_take(tm, eid, meta):
    tm.create_episode(None, episode_id=eid)
    if meta is not None:
        (tm.episode_dir(eid) / "metadata.json").write_text(json.dumps(meta))
    tm.register_episode(eid)


def test_a_clean_take_reports_no_issue(tm):
    _saved_take(tm, "20261125_150000", {"frame_count": 300, "imu_sample_count": 2000})
    assert tm.last_episode_issues() == ("20261125_150000", [])


def test_a_take_without_frames_or_imu_samples_reports_them(tm):
    _saved_take(tm, "20261125_150000", {"frame_count": 0, "imu_sample_count": 0})
    assert tm.last_episode_issues() == ("20261125_150000", ["camera", "imu"])


def test_a_take_without_metadata_reports_it(tm):
    _saved_take(tm, "20261125_150000", None)
    assert tm.last_episode_issues() == ("20261125_150000", ["metadata"])


def test_a_good_take_after_a_failed_start_replaces_the_failure(tm):
    tm.create_episode(None, episode_id="20261125_150000")
    tm.discard_pending_episode(start_failed=True)
    assert tm.last_episode_issues() == ("20261125_150000", ["start_failed"])

    _saved_take(tm, "20261125_150100", {"frame_count": 300, "imu_sample_count": 2000})
    assert tm.last_episode_issues() == ("20261125_150100", [])


def test_a_deleted_last_take_is_no_longer_reported(tm):
    _saved_take(tm, "20261125_150000", {"frame_count": 300, "imu_sample_count": 2000})
    tm.delete_episode("20261125_150000")
    assert tm.last_episode_issues() is None


# --- the relay puts both on the heartbeat --------------------------------------

async def test_the_heartbeat_carries_the_fault_and_the_last_take(monkeypatch):
    monkeypatch.setattr(relay_mod, "_HEARTBEAT_INTERVAL_S", 0.05)
    beats: list[dict] = []
    got_beat = asyncio.Event()

    async def register(request):
        return web.json_response({"status": "ok"})

    async def poll(request):
        await asyncio.sleep(0.05)
        return web.json_response({"commands": []})

    async def heartbeat(request):
        beats.append(dict(request.query))
        got_beat.set()
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_post("/api/devices/register", register)
    app.router.add_get("/api/devices/poll", poll)
    app.router.add_post("/api/devices/heartbeat", heartbeat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    client = relay_mod.RelayClient(
        f"http://127.0.0.1:{port}", lambda: "tok", "casquette-1",
        fault_provider=lambda: FAULT,
        last_episode_provider=lambda: ("20261125_155326", ["start_failed"]),
    )

    async def handler(cmd):
        return {"status": "ok"}

    task = asyncio.create_task(client.run(handler))
    try:
        await asyncio.wait_for(got_beat.wait(), timeout=3)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await runner.cleanup()

    b = beats[0]
    assert b["error"] == FAULT
    assert b["last_episode"] == "20261125_155326"
    assert b["last_issues"] == "start_failed"
