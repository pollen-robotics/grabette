"""/api/sync orchestrator tests.

Exercises the fan-out protocol in casquette.app.routers.sync by calling
the endpoint coroutines directly with a real EpisodeScheduler (over a
MockBackend) and mocking the peer HTTP surface with respx.

We assert the transaction semantics: preflight gates the start, a peer
failure rolls back already-committed peers, and the happy path schedules
locally after peers are committed.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import HTTPException

from casquette.app.routers.sync import sync_start, sync_stop
from casquette.config import settings
from casquette.scheduler import CaptureState

P1 = "http://p1.local:8000"
P2 = "http://p2.local:8000"


def _time_ok():
    return httpx.Response(
        200,
        json={"now_utc": "2026-01-01T00:00:00+00:00",
              "ntp_synchronized": True, "offset_us": 100},
    )


async def test_no_peers_schedules_local_only(scheduler, monkeypatch):
    monkeypatch.setattr(settings, "peers", "")
    resp = await sync_start(scheduler=scheduler)
    assert resp.local_episode_id
    assert resp.peers == []
    # LEAD_MS is in the future → local capture is scheduled, not recording.
    assert scheduler.state == CaptureState.SCHEDULED


async def test_preflight_failure_refuses_and_leaves_idle(scheduler, monkeypatch):
    monkeypatch.setattr(settings, "peers", f"p1={P1}")
    with respx.mock:
        respx.get(f"{P1}/api/system/time").mock(
            return_value=httpx.Response(
                200, json={"now_utc": "x", "ntp_synchronized": False}),
        )
        with pytest.raises(HTTPException) as ei:
            await sync_start(scheduler=scheduler)
    assert ei.value.status_code == 502
    assert ei.value.detail["phase"] == "preflight"
    assert scheduler.state == CaptureState.IDLE  # local never started


async def test_peer_start_failure_rolls_back_committed_peers(scheduler, monkeypatch):
    monkeypatch.setattr(settings, "peers", f"p1={P1},p2={P2}")
    with respx.mock:
        respx.get(f"{P1}/api/system/time").mock(return_value=_time_ok())
        respx.get(f"{P2}/api/system/time").mock(return_value=_time_ok())
        # p1 accepts the scheduled start, p2 rejects it.
        respx.post(f"{P1}/api/episodes/start").mock(
            return_value=httpx.Response(200, json={"episode_id": "p1-ep"}))
        respx.post(f"{P2}/api/episodes/start").mock(
            return_value=httpx.Response(500, text="boom"))
        # The rollback must POST stop to the peer that DID commit (p1).
        p1_stop = respx.post(f"{P1}/api/episodes/stop").mock(
            return_value=httpx.Response(200, json={"status": "cancelled"}))

        with pytest.raises(HTTPException) as ei:
            await sync_start(scheduler=scheduler)

        assert p1_stop.called  # p1 was rolled back
    assert ei.value.status_code == 502
    assert ei.value.detail["phase"] == "peer_start"
    assert "p1" in ei.value.detail["rolled_back"]
    assert scheduler.state == CaptureState.IDLE  # local refused


async def test_all_peers_ok_schedules_local(scheduler, monkeypatch):
    monkeypatch.setattr(settings, "peers", f"p1={P1}")
    with respx.mock:
        respx.get(f"{P1}/api/system/time").mock(return_value=_time_ok())
        respx.post(f"{P1}/api/episodes/start").mock(
            return_value=httpx.Response(200, json={"episode_id": "p1-ep"}))
        resp = await sync_start(scheduler=scheduler)
    assert scheduler.state == CaptureState.SCHEDULED
    assert resp.local_episode_id
    assert len(resp.peers) == 1
    assert resp.peers[0]["device_id"] == "p1"
    assert resp.peers[0]["episode_id"] == "p1-ep"


async def test_sync_stop_fans_out_to_peers(scheduler, monkeypatch):
    monkeypatch.setattr(settings, "peers", f"p1={P1}")
    await scheduler.start()  # local RECORDING
    with respx.mock:
        stop_route = respx.post(f"{P1}/api/episodes/stop").mock(
            return_value=httpx.Response(200, json={"status": "idle"}))
        resp = await sync_stop(scheduler=scheduler)
        assert stop_route.called
    assert resp.local is not None
    assert resp.peers[0]["device_id"] == "p1"
    assert resp.peers[0]["status"] == "ok"
    assert scheduler.state == CaptureState.IDLE
