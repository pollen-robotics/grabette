from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from grabette.app.dependencies import get_daemon
from grabette.daemon import Daemon

router = APIRouter(prefix="/api/state", tags=["state"])


@router.get("")
def get_state(daemon: Daemon = Depends(get_daemon)):
    # Cached poll-loop state (no blocking I2C) — this endpoint is polled often
    # by the dashboard, so a fresh read here would saturate the I2C bus.
    return daemon.latest_state()


@router.get("/history")
def get_state_history(cursor: int = 0, daemon: Daemon = Depends(get_daemon)):
    result = daemon.get_active_ring().get_since(cursor)
    result["gen"] = daemon.generation
    return result


@router.websocket("/ws")
async def state_ws(ws: WebSocket):
    await ws.accept()
    from grabette.app.main import get_daemon_instance
    try:
        while True:
            daemon = get_daemon_instance()
            if daemon and daemon.state.value == "running":
                # Cached state — NOT a fresh get_state(): a blocking I2C read
                # here runs on the event loop at 10Hz and stalls the whole server.
                await ws.send_json(daemon.latest_state().model_dump())
            await asyncio.sleep(0.1)  # 10Hz
    except WebSocketDisconnect:
        pass
