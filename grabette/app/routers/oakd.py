"""OAK-D runtime enable/disable.

The OAK-D draws non-trivial power; we keep it OFF by default at boot and let
the UI toggle it on demand. Toggling is refused while a capture is running
or teleop is active (mirrors the teleop router's 409 pattern).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from grabette.app.dependencies import get_backend
from grabette.backend.base import Backend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/oakd", tags=["oakd"])


def _status(backend: Backend) -> dict:
    enabled = getattr(backend, "is_oakd_enabled", False)
    initialized = getattr(backend, "is_oakd_initialized", False)
    return {
        "supported": hasattr(backend, "set_oakd_enabled"),
        "enabled": bool(enabled),
        "initialized": bool(initialized),
    }


@router.get("/status")
def oakd_status(backend: Backend = Depends(get_backend)):
    return _status(backend)


@router.post("/enable")
async def oakd_enable(backend: Backend = Depends(get_backend)):
    if not hasattr(backend, "set_oakd_enabled"):
        raise HTTPException(status_code=501, detail="backend has no OAK-D")
    try:
        await backend.set_oakd_enabled(True)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _status(backend)


@router.post("/disable")
async def oakd_disable(backend: Backend = Depends(get_backend)):
    if not hasattr(backend, "set_oakd_enabled"):
        raise HTTPException(status_code=501, detail="backend has no OAK-D")
    try:
        await backend.set_oakd_enabled(False)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return _status(backend)
