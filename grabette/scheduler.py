"""Episode scheduler — wraps backend start/stop with a state machine that
supports both immediate and future-anchored starts.

States:
    IDLE      → nothing scheduled, not capturing
    SCHEDULED → asyncio task waiting until target UTC, then will start
    RECORDING → backend.is_capturing == True

Transitions:
    IDLE      --start(now or T₀)--> SCHEDULED or RECORDING
    SCHEDULED --T₀ fires----------> RECORDING
    SCHEDULED --stop()------------> IDLE  (task cancelled, episode dir deleted)
    RECORDING --stop()------------> IDLE  (backend.stop_capture)

Only one episode at a time per device. start() refuses with RuntimeError when
state != IDLE; stop() refuses when state == IDLE.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from grabette.backend.base import Backend
from grabette.session import SessionManager

logger = logging.getLogger(__name__)


class CaptureState(str, Enum):
    IDLE = "idle"
    SCHEDULED = "scheduled"
    RECORDING = "recording"


class EpisodeScheduler:
    def __init__(self, backend: Backend, sm: SessionManager) -> None:
        self._backend = backend
        self._sm = sm
        self._scheduled_task: asyncio.Task | None = None
        self._scheduled_at_utc: datetime | None = None
        self._scheduled_episode_id: str | None = None

    @property
    def state(self) -> CaptureState:
        if self._backend.is_capturing:
            return CaptureState.RECORDING
        if self._scheduled_task is not None and not self._scheduled_task.done():
            return CaptureState.SCHEDULED
        return CaptureState.IDLE

    @property
    def scheduled_at_utc(self) -> datetime | None:
        return self._scheduled_at_utc

    @property
    def scheduled_episode_id(self) -> str | None:
        return self._scheduled_episode_id

    async def start(self, start_at_utc: datetime | None = None) -> str:
        """Start now or schedule at start_at_utc. Returns the episode_id.

        Raises:
            RuntimeError if not in IDLE state (caller should map to HTTP 409).
            ValueError if start_at_utc is in the past (caller should map to 400).
        """
        if self.state != CaptureState.IDLE:
            raise RuntimeError(
                f"Cannot start: state is {self.state.value}"
            )

        if start_at_utc is not None:
            if start_at_utc.tzinfo is None:
                # Treat naive datetimes as UTC explicitly rather than guessing.
                start_at_utc = start_at_utc.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            if start_at_utc <= now:
                raise ValueError(
                    f"start_at_utc is in the past: "
                    f"{start_at_utc.isoformat()} (now {now.isoformat()})"
                )

        # Create the episode dir up front so the returned id is stable
        # regardless of the start path.
        episode_id = self._sm.create_episode()
        episode_dir = self._sm.episode_dir(episode_id)

        if start_at_utc is None:
            await self._backend.start_capture(episode_dir)
            self._scheduled_episode_id = episode_id
            return episode_id

        self._scheduled_episode_id = episode_id
        self._scheduled_at_utc = start_at_utc
        self._scheduled_task = asyncio.create_task(
            self._wait_and_start(start_at_utc, episode_dir),
            name=f"scheduled-start-{episode_id}",
        )
        return episode_id

    async def _wait_and_start(
        self, target_utc: datetime, episode_dir: Path,
    ) -> None:
        try:
            wait_s = (target_utc - datetime.now(timezone.utc)).total_seconds()
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            await self._backend.start_capture(episode_dir)
            logger.info(
                "Scheduled start fired at %s (actual skew %+.3f ms)",
                target_utc.isoformat(),
                (datetime.now(timezone.utc) - target_utc).total_seconds() * 1000,
            )
        except asyncio.CancelledError:
            logger.info("Scheduled start cancelled before T0")
            raise
        except Exception:
            logger.exception("Scheduled start failed; clearing scheduled state")
            self._scheduled_at_utc = None
            self._scheduled_episode_id = None

    async def stop(self):
        """Cancel a scheduled start or stop an active recording.

        Returns the CaptureStatus from backend.stop_capture if there was a
        recording to stop, or None if a scheduled task was cancelled.
        Raises RuntimeError if state == IDLE (caller should map to 409).
        """
        s = self.state
        if s == CaptureState.IDLE:
            raise RuntimeError("Not scheduled or capturing")

        if s == CaptureState.SCHEDULED:
            assert self._scheduled_task is not None
            self._scheduled_task.cancel()
            try:
                await self._scheduled_task
            except asyncio.CancelledError:
                pass
            cancelled_episode_id = self._scheduled_episode_id
            self._scheduled_task = None
            self._scheduled_at_utc = None
            self._scheduled_episode_id = None
            # Remove the pre-created (empty) episode dir.
            if cancelled_episode_id:
                try:
                    self._sm.delete_episode(cancelled_episode_id)
                except Exception:
                    logger.debug(
                        "Could not clean up cancelled episode %s",
                        cancelled_episode_id, exc_info=True,
                    )
            return None

        # state == RECORDING
        status = await self._backend.stop_capture()
        self._scheduled_task = None
        self._scheduled_at_utc = None
        self._scheduled_episode_id = None
        return status
