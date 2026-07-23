"""Episode scheduler — wraps backend start/stop with a state machine that
supports both immediate and future-anchored starts.

States:
    IDLE      → nothing scheduled, not capturing
    SCHEDULED → asyncio task waiting until target UTC
    STARTING  → past T₀, inside backend.start_capture (slow on hardware
                with heavy init, e.g. OAK-D ~4s). Important to distinguish
                because backend.is_capturing is still False here, but the
                scheduled task can no longer be safely cancelled — that
                would interrupt hardware init mid-flight.
    RECORDING → backend.is_capturing == True

Transitions:
    IDLE      --start(now or T₀)--> SCHEDULED or STARTING
    SCHEDULED --T₀ fires----------> STARTING
    STARTING  --start_capture done> RECORDING (or IDLE on failure)
    SCHEDULED --stop()------------> IDLE  (task cancelled, dir deleted)
    STARTING  --stop()------------> awaits start to finish, then stops
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

from casquette.backend.base import Backend
from casquette.session import SessionManager

logger = logging.getLogger(__name__)


class CaptureState(str, Enum):
    IDLE = "idle"
    SCHEDULED = "scheduled"
    STARTING = "starting"
    RECORDING = "recording"


class EpisodeScheduler:
    def __init__(self, backend: Backend, sm: SessionManager) -> None:
        self._backend = backend
        self._sm = sm
        self._scheduled_task: asyncio.Task | None = None
        self._scheduled_at_utc: datetime | None = None
        self._scheduled_episode_id: str | None = None
        # Peer list for the current scheduled episode (populated by the
        # /api/sync/start orchestrator). Includes self when sync-driven.
        # Stored here so _wait_and_start can attach the full rig topology
        # to backend.set_sync_metadata after start_capture succeeds.
        self._scheduled_peers: list[dict] = []
        # True while backend.start_capture is in progress (past T₀, before
        # is_capturing flips). Used to distinguish SCHEDULED (safe to
        # cancel) from STARTING (must wait for start to complete).
        self._starting: bool = False

    @property
    def state(self) -> CaptureState:
        if self._backend.is_capturing:
            return CaptureState.RECORDING
        if self._starting:
            return CaptureState.STARTING
        if self._scheduled_task is not None and not self._scheduled_task.done():
            return CaptureState.SCHEDULED
        return CaptureState.IDLE

    @property
    def scheduled_at_utc(self) -> datetime | None:
        return self._scheduled_at_utc

    @property
    def scheduled_episode_id(self) -> str | None:
        return self._scheduled_episode_id

    async def start(
        self,
        start_at_utc: datetime | None = None,
        peers: list[dict] | None = None,
    ) -> str:
        """Start now or schedule at start_at_utc. Returns the episode_id.

        Args:
            start_at_utc: UTC instant to start at. None = start immediately.
            peers: full rig topology for sync-driven starts (each entry is a
                dict with at least 'device_id', optionally 'url' and 'role').
                Stored and attached to metadata.json at stop time. Empty /
                None for local-only captures.

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

        # Clear any stale sync metadata from a previous capture. The
        # set_sync_metadata in _wait_and_start re-populates it after
        # start_capture succeeds; this baseline reset ensures local-only
        # captures don't accidentally inherit a previous sync episode's data.
        self._backend.set_sync_metadata({})

        # Create the episode dir up front so the returned id is stable
        # regardless of the start path.
        episode_id = self._sm.create_episode()
        episode_dir = self._sm.episode_dir(episode_id)
        self._scheduled_peers = list(peers or [])

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
            # Measure the sync-precision skew at sleep-end, BEFORE the
            # potentially-slow start_capture (~6 s for OAK-D init). The
            # earlier formulation logged the skew AFTER start_capture,
            # which conflated NTP precision with hardware init time.
            sleep_end_skew_ms = (
                datetime.now(timezone.utc) - target_utc
            ).total_seconds() * 1000
            logger.info(
                "T0 reached at %s (skew %+.3f ms)",
                target_utc.isoformat(), sleep_end_skew_ms,
            )
            # From here on, we are in STARTING: cancellation could damage
            # hardware init in progress. The flag is cleared in `finally`
            # regardless of success/failure/cancellation.
            self._starting = True
            try:
                t0 = datetime.now(timezone.utc)
                await self._backend.start_capture(episode_dir)
                init_ms = (
                    datetime.now(timezone.utc) - t0
                ).total_seconds() * 1000
                logger.info(
                    "start_capture completed (took %.0f ms)", init_ms,
                )
                # Attach sync metadata for stop_capture to fold into
                # metadata.json. Per-device file is self-contained:
                # workstation analysis can group multi-device episodes
                # by matching scheduled_start_utc across files.
                self._backend.set_sync_metadata({
                    "scheduled_start_utc": target_utc.isoformat(),
                    "sleep_end_skew_ms": round(sleep_end_skew_ms, 3),
                    "start_capture_ms": round(init_ms, 0),
                    "peers": list(self._scheduled_peers),
                })
            finally:
                self._starting = False
        except asyncio.CancelledError:
            # Distinguish "before T₀" (pure schedule cancel) from
            # "during start_capture" (more disruptive) for log clarity.
            phase = "during start_capture" if self._starting else "before T0"
            logger.info("Scheduled start cancelled %s", phase)
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

        When state is STARTING (hardware init in progress past T₀), waits
        for start to complete before stopping, with a bounded timeout — a
        stop() arriving mid-init must NOT interrupt the in-flight
        start_capture (could leave hardware in an inconsistent state).
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
            self._clear_scheduled()
            if cancelled_episode_id:
                try:
                    self._sm.delete_episode(cancelled_episode_id)
                except Exception:
                    logger.debug(
                        "Could not clean up cancelled episode %s",
                        cancelled_episode_id, exc_info=True,
                    )
            return None

        if s == CaptureState.STARTING:
            # Wait for start_capture to finish (success or fail) before
            # we can stop. Timeout long enough for slow inits (OAK-D = ~5s).
            assert self._scheduled_task is not None
            try:
                await asyncio.wait_for(self._scheduled_task, timeout=15.0)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "start_capture is still running after 15s; refusing to stop"
                )
            except Exception:
                # _wait_and_start already logs and clears state on failure.
                self._clear_scheduled()
                return None
            # If we got here, start completed — fall through to RECORDING.
            if not self._backend.is_capturing:
                # start_capture returned but didn't actually start (e.g.
                # silently failed before flipping is_capturing).
                self._clear_scheduled()
                return None

        # state == RECORDING (either originally or just transitioned).
        status = await self._backend.stop_capture()
        self._clear_scheduled()
        return status

    def _clear_scheduled(self) -> None:
        self._scheduled_task = None
        self._scheduled_at_utc = None
        self._scheduled_episode_id = None
