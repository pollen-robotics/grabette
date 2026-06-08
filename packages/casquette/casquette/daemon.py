from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from enum import Enum

from casquette.backend.base import Backend

logger = logging.getLogger(__name__)


class SampleRing:
    """Thread-safe ring buffer with cursor-based reads (multi-consumer safe)."""

    def __init__(self, maxlen: int = 500) -> None:
        self._imu: deque[tuple[int, dict]] = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()

    def push_state(self, state) -> None:
        with self._lock:
            self._seq += 1
            seq = self._seq
            if state.imu is not None:
                s = state.imu
                self._imu.append((seq, {"t": s.timestamp_ms, "a": list(s.accel), "g": list(s.gyro)}))

    def push_raw(self, imu: dict | None = None) -> None:
        """Push pre-formatted dict directly."""
        with self._lock:
            self._seq += 1
            seq = self._seq
            if imu is not None:
                self._imu.append((seq, imu))

    def get_since(self, cursor: int = 0) -> dict:
        """Return samples with seq > cursor."""
        with self._lock:
            imu = [s for seq, s in self._imu if seq > cursor]
            seq = self._seq
        return {"imu": imu, "cursor": seq}


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
        self.sample_ring = SampleRing()
        self._poll_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.state not in (DaemonState.NOT_INITIALIZED, DaemonState.STOPPED, DaemonState.ERROR):
            logger.warning("Cannot start daemon from state %s", self.state)
            return
        self.state = DaemonState.STARTING
        self._error = None
        try:
            await self.backend.start()
            self._poll_task = asyncio.create_task(self._poll_loop())
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
            if self._poll_task is not None:
                self._poll_task.cancel()
                try:
                    await self._poll_task
                except asyncio.CancelledError:
                    pass
                self._poll_task = None
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

    async def _poll_loop(self) -> None:
        """Poll backend at ~50Hz and push samples into the ring buffer."""
        while True:
            try:
                state = self.backend.get_state()
                self.sample_ring.push_state(state)
            except Exception:
                logger.debug("Poll loop sample error", exc_info=True)
            await asyncio.sleep(0.02)  # 50Hz

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
