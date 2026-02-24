from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from grabette.app.dependencies import get_backend, get_daemon
from grabette.backend.base import Backend
from grabette.daemon import Daemon

router = APIRouter(prefix="/api/state", tags=["state"])


@router.get("")
def get_state(backend: Backend = Depends(get_backend)):
    return backend.get_state()


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
                state = daemon.backend.get_state()
                await ws.send_json(state.model_dump())
            await asyncio.sleep(0.1)  # 10Hz
    except WebSocketDisconnect:
        pass
