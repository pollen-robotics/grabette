from __future__ import annotations

from fastapi import HTTPException

from casquette.backend.base import Backend
from casquette.daemon import Daemon, DaemonState


def get_daemon() -> Daemon:
    from casquette.app.main import get_daemon_instance

    daemon = get_daemon_instance()
    if daemon is None:
        raise HTTPException(status_code=503, detail="Daemon not initialized")
    return daemon


def get_backend() -> Backend:
    daemon = get_daemon()
    if daemon.state != DaemonState.RUNNING:
        raise HTTPException(status_code=503, detail=f"Daemon not ready (state: {daemon.state.value})")
    return daemon.backend
