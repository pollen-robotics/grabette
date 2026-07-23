from __future__ import annotations

from fastapi import HTTPException

from casquette.backend.base import Backend
from casquette.daemon import Daemon, DaemonState
from casquette.scheduler import EpisodeScheduler


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


# Lazily-instantiated, single-process singleton. Created on first use
# after the daemon is RUNNING (the scheduler needs the live backend).
_scheduler: EpisodeScheduler | None = None


def get_scheduler() -> EpisodeScheduler:
    global _scheduler
    if _scheduler is None:
        from casquette.app.routers.sessions import get_session_manager
        backend = get_backend()
        _scheduler = EpisodeScheduler(backend, get_session_manager())
    return _scheduler
