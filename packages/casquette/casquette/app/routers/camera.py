from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from casquette.app.dependencies import get_backend
from casquette.backend.base import Backend


class _ExposureBody(BaseModel):
    us: int = Field(
        ...,
        ge=0,
        description=(
            "Exposure time in microseconds. > 0 fixes the shutter (auto-gain "
            "still active); 0 attempts to restore libcamera auto-exposure "
            "(may need a daemon restart to fully take effect)."
        ),
    )

router = APIRouter(prefix="/api/camera", tags=["camera"])

# MJPEG stream constants. Boundary is arbitrary as long as it's stable
# across the response and matches the multipart/x-mixed-replace declaration.
_MJPEG_BOUNDARY = b"frame"
_MJPEG_FPS = 15.0


@router.get("/exposure")
def get_exposure(backend: Backend = Depends(get_backend)):
    return {"us": backend.get_camera_exposure_us()}


@router.post("/exposure")
def set_exposure(body: _ExposureBody, backend: Backend = Depends(get_backend)):
    try:
        applied = backend.set_camera_exposure_us(body.us)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"us": applied}


@router.get("/snapshot")
def camera_snapshot(backend: Backend = Depends(get_backend)):
    frame = backend.get_frame_jpeg()
    if frame is None:
        return Response(status_code=503, content="Camera not available")
    content_type = "image/bmp" if frame[:2] == b"BM" else "image/jpeg"
    return Response(content=frame, media_type=content_type)


@router.get("/stream")
async def camera_stream():
    """Browser-renderable MJPEG stream.

    Open `http://<host>:8001/api/camera/stream` directly in a browser and
    you'll see a live feed — works in Chrome/Firefox/Safari without any
    extra page or JavaScript. Useful for FOV/exposure experimentation.

    multipart/x-mixed-replace is rate-limited to _MJPEG_FPS by an
    asyncio.sleep to keep the Pi Zero 2W's encode + network load
    bounded. If the daemon isn't running, the stream just yields
    nothing (browser shows a blank tab) — no error frames sent.
    """
    from casquette.app.main import get_daemon_instance

    async def gen():
        dt = 1.0 / _MJPEG_FPS
        while True:
            daemon = get_daemon_instance()
            if daemon and daemon.state.value == "running":
                frame = daemon.backend.get_frame_jpeg()
                if frame:
                    yield b"--" + _MJPEG_BOUNDARY + b"\r\n"
                    yield b"Content-Type: image/jpeg\r\n"
                    yield f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    yield frame
                    yield b"\r\n"
            await asyncio.sleep(dt)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY.decode()}",
    )


@router.websocket("/ws")
async def camera_ws(ws: WebSocket):
    await ws.accept()
    from casquette.app.main import get_daemon_instance
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
