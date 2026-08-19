"""Fleet command-dispatcher tests (_handle_relay_command).

Exercises the recording subset casquette handles as a pure fleet peer:
get_state, cancel_dataset, immediate + synchronized (T0) start/stop, and the
too-late-start guard — all against the mock backend + a tmp task store.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from casquette.app.main import _handle_relay_command


def _utc_iso(offset_s: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


async def test_get_state_reports_running(daemon, tm):
    r = await _handle_relay_command({"type": "get_state", "id": "c1"})
    assert r["status"] == "ok"
    assert r["state"]["state"] == "running"


async def test_cancel_dataset_marks_ids(daemon, tm):
    r = await _handle_relay_command(
        {"type": "cancel_dataset", "id": "c2", "args": {"command_ids": ["j1"]}}
    )
    assert r["status"] == "ok"
    assert r["cancelled"] == ["j1"]


async def test_immediate_start_stop_records_episode(daemon, tm):
    r = await _handle_relay_command(
        {"type": "start_capture", "id": "c3", "args": {"task_name": "pick"}}
    )
    assert r["status"] == "ok"
    episode_id = r["episode_id"]
    assert daemon.backend.is_capturing

    r2 = await _handle_relay_command({"type": "stop_capture", "id": "c4", "args": {}})
    assert r2["status"] == "ok"
    assert not daemon.backend.is_capturing

    # The episode is filed under the "pick" task in the record store.
    tasks = {t["name"]: t for t in tm.report_tasks()}
    assert "pick" in tasks
    assert episode_id in tm.episode_dir(episode_id).name
    assert tm.episode_dir(episode_id).exists()


async def test_start_rejected_when_already_capturing(daemon, tm):
    await _handle_relay_command(
        {"type": "start_capture", "id": "c5", "args": {"task_name": "pick"}}
    )
    r = await _handle_relay_command(
        {"type": "start_capture", "id": "c6", "args": {"task_name": "pick"}}
    )
    assert r["status"] == "error"
    await _handle_relay_command({"type": "stop_capture", "id": "c7", "args": {}})


async def test_scheduled_start_shares_t0_episode_id_and_fires(daemon, tm):
    t0 = _utc_iso(0.3)
    r = await _handle_relay_command(
        {"type": "start_capture", "id": "c8",
         "args": {"task_name": "pick", "start_at_utc": t0}}
    )
    assert r["status"] == "scheduled"
    # episode id is derived from the shared T0 (same on every device), not local
    # wall-clock creation time.
    assert r["episode_id"] == datetime.fromisoformat(t0).astimezone(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    await asyncio.sleep(0.6)  # wait past T0
    assert daemon.backend.is_capturing
    r2 = await _handle_relay_command({"type": "stop_capture", "id": "c9", "args": {}})
    assert r2["status"] == "ok"


async def test_start_too_late_is_refused(daemon, tm):
    r = await _handle_relay_command(
        {"type": "start_capture", "id": "c10",
         "args": {"task_name": "pick", "start_at_utc": _utc_iso(-5)}}
    )
    assert r["status"] == "error"
    assert "late" in r["message"]
    assert not daemon.backend.is_capturing


async def test_unknown_command(daemon, tm):
    r = await _handle_relay_command({"type": "frobnicate", "id": "c11"})
    assert r["status"] == "error"
    assert "unknown" in r["message"]
