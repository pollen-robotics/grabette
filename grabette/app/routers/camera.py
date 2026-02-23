from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from grabette.app.dependencies import get_backend
from grabette.backend.base import Backend

router = APIRouter(prefix="/api/camera", tags=["camera"])


@router.get("/snapshot")
def camera_snapshot(backend: Backend = Depends(get_backend)):
    frame = backend.get_frame_jpeg()
    if frame is None:
        return Response(status_code=503, content="Camera not available")
    # Detect format from magic bytes
    content_type = "image/bmp" if frame[:2] == b"BM" else "image/jpeg"
    return Response(content=frame, media_type=content_type)


@router.websocket("/ws")
async def camera_ws(ws: WebSocket):
    await ws.accept()
    from grabette.app.main import get_daemon_instance
    try:
        while True:
            daemon = get_daemon_instance()
            if daemon and daemon.state.value == "running":
                frame = daemon.backend.get_frame_jpeg()
                if frame:
                    await ws.send_bytes(frame)
            await asyncio.sleep(1 / 15)  # ~15fps
    except WebSocketDisconnect:
        pass
