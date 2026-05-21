"""Teleop mode routes.

Mode-switching:
    POST /api/teleop/start  → daemon enters teleop mode (live VIO on OAK).
                              Mutually exclusive with recording. Fails 409 if
                              a capture is in progress.
    POST /api/teleop/stop   → exits teleop mode, re-initializes the
                              recording-mode OAK pipeline.
    GET  /api/teleop/status → current state + framerate stats.
    WS   /api/teleop/stream → JSON delta stream at ~30 Hz.

JSON delta payload (one message per tick):
    {"t": 12.345, "send": true, "lost": false,
     "dx": 0.001, "dy": 0.0, "dz": 0.0,
     "dqx": 0.0, "dqy": 0.0, "dqz": 0.0, "dqw": 1.0}

The `send` flag is reserved for Phase 2.4 (button gating). For now it's
always True while teleop is active.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from grabette.app.dependencies import get_backend
from grabette.backend.base import Backend

router = APIRouter(prefix="/api/teleop", tags=["teleop"])

STREAM_RATE_HZ = 30.0
_STREAM_DT = 1.0 / STREAM_RATE_HZ


@router.post("/start")
async def start_teleop(backend: Backend = Depends(get_backend)):
    try:
        await backend.start_teleop()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    return {"active": True}


@router.post("/stop")
async def stop_teleop(backend: Backend = Depends(get_backend)):
    try:
        await backend.stop_teleop()
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    return {"active": False}


@router.get("/status")
def teleop_status(backend: Backend = Depends(get_backend)):
    return {
        "active": backend.is_teleop_active,
        "stats": backend.get_teleop_stats(),
        "pose": backend.get_teleop_pose(),
    }


@router.websocket("/stream")
async def teleop_stream(ws: WebSocket):
    """JSON delta stream at a fixed STREAM_RATE_HZ.

    Emits the latest computed delta on every tick — repeats the last value
    if SLAM hasn't produced a new pose since the previous tick (steady rate
    matters more to the robot consumer than per-pose freshness).
    """
    await ws.accept()
    # Lazy import to avoid circular dep (router → main → router)
    from grabette.app.main import get_daemon_instance

    try:
        while True:
            daemon = get_daemon_instance()
            backend: Backend | None = daemon.backend if daemon else None
            if backend is not None and backend.is_teleop_active:
                d = backend.get_teleop_delta()
                if d is None:
                    msg = {
                        "t": 0.0, "send": False, "lost": True,
                        "dx": 0.0, "dy": 0.0, "dz": 0.0,
                        "dqx": 0.0, "dqy": 0.0, "dqz": 0.0, "dqw": 1.0,
                    }
                else:
                    msg = {
                        "t": d["t_host"],
                        "send": True,        # button gating goes here in Phase 2.4
                        "lost": False,
                        "dx": d["dx"], "dy": d["dy"], "dz": d["dz"],
                        "dqx": d["dqx"], "dqy": d["dqy"],
                        "dqz": d["dqz"], "dqw": d["dqw"],
                    }
                await ws.send_json(msg)
            await asyncio.sleep(_STREAM_DT)
    except WebSocketDisconnect:
        pass
