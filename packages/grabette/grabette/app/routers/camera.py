from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from grabette.app.dependencies import get_backend
from grabette.backend.base import Backend

router = APIRouter(prefix="/api/camera", tags=["camera"])


@router.get("/status")
def camera_status(backend: Backend = Depends(get_backend)):
    """Lightweight RGB camera connection check (stays True during capture)."""
    return {"connected": bool(getattr(backend, "is_camera_connected", False))}


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


@router.get("/depth")
def depth_snapshot(backend: Backend = Depends(get_backend)):
    """Colorized OAK-D depth frame (turbo colormap, 0.2-3m)."""
    frame = backend.get_depth_jpeg()
    if frame is None:
        return Response(status_code=503, content="Depth not available")
    return Response(content=frame, media_type="image/jpeg")


@router.websocket("/depth_ws")
async def depth_ws(ws: WebSocket):
    """WebSocket stream of colorized OAK-D depth frames at ~15fps."""
    await ws.accept()
    from grabette.app.main import get_daemon_instance
    try:
        while True:
            daemon = get_daemon_instance()
            if daemon and daemon.state.value == "running":
                frame = daemon.backend.get_depth_jpeg()
                if frame:
                    await ws.send_bytes(frame)
            await asyncio.sleep(1 / 15)
    except WebSocketDisconnect:
        pass
