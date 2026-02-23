from __future__ import annotations

from fastapi import APIRouter, Depends

from grabette.app.dependencies import get_daemon
from grabette.daemon import Daemon

router = APIRouter(prefix="/api/daemon", tags=["daemon"])


@router.get("/status")
def daemon_status(daemon: Daemon = Depends(get_daemon)):
    return daemon.status


@router.post("/restart")
async def daemon_restart(daemon: Daemon = Depends(get_daemon)):
    await daemon.restart()
    return daemon.status
