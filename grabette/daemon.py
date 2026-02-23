from __future__ import annotations

import asyncio
import logging
from enum import Enum

from grabette.backend.base import Backend

logger = logging.getLogger(__name__)


class DaemonState(str, Enum):
    NOT_INITIALIZED = "not_initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class Daemon:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.state = DaemonState.NOT_INITIALIZED
        self._error: str | None = None

    async def start(self) -> None:
        if self.state not in (DaemonState.NOT_INITIALIZED, DaemonState.STOPPED, DaemonState.ERROR):
            logger.warning("Cannot start daemon from state %s", self.state)
            return
        self.state = DaemonState.STARTING
        self._error = None
        try:
            await self.backend.start()
            self.state = DaemonState.RUNNING
            logger.info("Daemon started with backend %s", type(self.backend).__name__)
        except Exception as exc:
            self._error = str(exc)
            self.state = DaemonState.ERROR
            logger.exception("Failed to start daemon")

    async def stop(self) -> None:
        if self.state != DaemonState.RUNNING:
            logger.warning("Cannot stop daemon from state %s", self.state)
            return
        self.state = DaemonState.STOPPING
        try:
            await self.backend.stop()
            self.state = DaemonState.STOPPED
            logger.info("Daemon stopped")
        except Exception as exc:
            self._error = str(exc)
            self.state = DaemonState.ERROR
            logger.exception("Failed to stop daemon")

    async def restart(self) -> None:
        if self.state == DaemonState.RUNNING:
            await self.stop()
        await asyncio.sleep(0.1)
        await self.start()

    @property
    def status(self) -> dict:
        result = {
            "state": self.state.value,
            "backend": type(self.backend).__name__,
            "error": self._error,
        }
        if self.state == DaemonState.RUNNING:
            result["sensor"] = self.backend.get_state().model_dump()
        return result
