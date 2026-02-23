from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from grabette.app.dependencies import get_backend
from grabette.backend.base import Backend

router = APIRouter(prefix="/api/state", tags=["state"])


@router.get("")
def get_state(backend: Backend = Depends(get_backend)):
    return backend.get_state()


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
