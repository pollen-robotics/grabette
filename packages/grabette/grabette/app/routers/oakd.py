"""Depth camera runtime enable/disable.

Serves whichever camera `depth_camera` selected — an OAK-D SR or an Orbbec
Gemini 305. The route prefix stays /api/oakd: renaming it would break the UI
client and any external caller, and is a separate change with a compat shim.

The camera draws non-trivial power; we keep it OFF by default at boot and let
the UI toggle it on demand. Toggling is refused while a capture is running
or teleop is active (mirrors the teleop router's 409 pattern).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from grabette.app.dependencies import get_backend
from grabette.backend.base import Backend
from grabette.hardware.depth_camera import display_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/oakd", tags=["oakd"])


def _status(backend: Backend) -> dict:
    enabled = getattr(backend, "is_oakd_enabled", False)
    initialized = getattr(backend, "is_oakd_initialized", False)
    initializing = getattr(backend, "is_oakd_initializing", False)
    # `model` is None on backends that cannot say (the mock). `label` is always
    # a usable string so consumers never have to fall back themselves.
    model = getattr(backend, "depth_camera_model", None)
    return {
        "supported": hasattr(backend, "set_oakd_enabled"),
        "enabled": bool(enabled),
        "initialized": bool(initialized),
        "initializing": bool(initializing),
        "model": model,
        "label": display_name(model),
    }


@router.get("/status")
def oakd_status(backend: Backend = Depends(get_backend)):
    return _status(backend)


@router.post("/enable")
async def oakd_enable(backend: Backend = Depends(get_backend)):
    if not hasattr(backend, "set_oakd_enabled"):
        raise HTTPException(status_code=501, detail="backend has no depth camera")
    try:
        await backend.set_oakd_enabled(True)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _status(backend)


@router.post("/disable")
async def oakd_disable(backend: Backend = Depends(get_backend)):
    if not hasattr(backend, "set_oakd_enabled"):
        raise HTTPException(status_code=501, detail="backend has no depth camera")
    try:
        await backend.set_oakd_enabled(False)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _status(backend)
