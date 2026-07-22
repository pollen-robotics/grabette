"""Schedules a capture start at a future UTC instant, for synchronized
multi-device group recordings.

Used by every local start trigger — the physical button (button_listener.py),
the local UI (app/routers/sessions.py), and fleet-dispatched commands
(app/main.py) — so a device grouped in grabette-fleet starts in lockstep with
its peers regardless of which one actually triggered the recording. Each
device waits out T0 on its own NTP-disciplined clock, so the round-trip
latency to discover/notify peers only delays when T0 is picked, not how
tightly the group starts once it fires.

A single process-wide instance (get_capture_scheduler()) since there's only
ever one capture — scheduled or running — per device.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class CaptureScheduler:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._start_at_utc: datetime | None = None
        # True once T0 has fired and backend.start_capture is in flight.
        # Distinguishes "safe to cancel" (still waiting) from "must let it
        # finish" (hardware init in progress) when a stop races the start.
        self._starting: bool = False

    def is_scheduled(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def is_starting(self) -> bool:
        return self._starting

    @property
    def scheduled_start_utc(self) -> datetime | None:
        return self._start_at_utc if self.is_scheduled() else None

    async def schedule(self, backend, sm, episode_dir: Path, start_at_utc: datetime) -> None:
        """Start a background wait-then-start task for a synchronized start."""
        self._start_at_utc = start_at_utc
        self._task = asyncio.create_task(
            self._wait_and_start(backend, sm, episode_dir, start_at_utc),
        )

    async def _wait_and_start(self, backend, sm, episode_dir: Path, target_utc: datetime) -> None:
        try:
            # Warm the hardware DURING the lead window (before T0), so the slow,
            # variable OAK-D bring-up doesn't sit between T0 and the first frame
            # — that variance is what makes two devices' recordings drift apart
            # by ~1s. At T0, start_capture then only starts the recording clock.
            # Best-effort: a warmup failure shouldn't cancel the start.
            try:
                await backend.prepare_capture()
            except Exception:
                logger.warning("prepare_capture (pre-T0 warmup) failed", exc_info=True)
            # Recompute the wait AFTER warmup — warmup ate part of the lead.
            wait_s = (target_utc - datetime.now(timezone.utc)).total_seconds()
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            self._starting = True
            try:
                wake = datetime.now(timezone.utc)
                await backend.start_capture(episode_dir)
                started = datetime.now(timezone.utc)
                logger.info("Scheduled start fired (target %s)", target_utc.isoformat())
                # Record multi-device sync info into the episode's metadata.json
                # (folded in at stop): the shared target lets a workstation PAIR
                # the per-device episodes; capture_started_utc lets it convert
                # each stream to absolute UTC and align them. wake_skew/start_ms
                # expose the residual timing for diagnostics.
                backend.set_sync_metadata({
                    "scheduled_start_utc": target_utc.isoformat(),
                    "capture_started_utc": started.isoformat(),
                    "wake_skew_ms": round((wake - target_utc).total_seconds() * 1000, 1),
                    "start_capture_ms": round((started - wake).total_seconds() * 1000, 1),
                })
            finally:
                self._starting = False
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Scheduled start failed; discarding pending episode")
            sm.discard_pending_episode()
        finally:
            self._task = None
            self._start_at_utc = None

    async def cancel_or_wait(self, backend) -> str:
        """Resolve a stop request against any pending scheduled start.

        Returns:
            "cancelled" — a not-yet-fired start was aborted; caller must
                discard the pending episode and must NOT call stop_capture.
            "ran" — nothing was scheduled, or it already started (or
                finished starting) by the time we checked; caller should
                proceed with a normal stop_capture if backend.is_capturing.

        Raises RuntimeError if start_capture is still running 15s after T0
        (refuses to interrupt hardware init mid-flight).
        """
        if not self.is_scheduled():
            return "ran"
        if self._starting:
            try:
                await asyncio.wait_for(self._task, timeout=15.0)
            except asyncio.TimeoutError:
                raise RuntimeError("start_capture still running after 15s; refusing to stop")
            return "ran"
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        return "cancelled"


_scheduler = CaptureScheduler()


def get_capture_scheduler() -> CaptureScheduler:
    return _scheduler
