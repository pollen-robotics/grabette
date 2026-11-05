"""capture_started_utc is what aligns the devices of a synchronized recording.

It must be the streams' t=0, not the end of start_capture, which keeps running
past t=0 (camera start) by a different amount on each device.
"""
import asyncio
from datetime import datetime, timedelta, timezone

from casquette.fleet.capture_scheduler import CaptureScheduler
from casquette.hardware.sync import SyncManager


class _Backend:
    def __init__(self, t0):
        self.recording_started_utc = None
        self._t0 = t0
        self.sync_meta = None

    async def prepare_capture(self):
        pass

    async def start_capture(self, episode_dir):
        self.recording_started_utc = self._t0
        await asyncio.sleep(0.1)  # camera start... after t=0

    def set_sync_metadata(self, meta):
        self.sync_meta = meta


def _run_scheduled_start(backend):
    async def _go():
        sched = CaptureScheduler()
        await sched.schedule(backend, None, None, datetime.now(timezone.utc))
        await sched._task

    asyncio.run(_go())
    return backend.sync_meta


def test_capture_started_utc_is_the_streams_t0():
    t0 = datetime.now(timezone.utc) + timedelta(milliseconds=2)
    meta = _run_scheduled_start(_Backend(t0))
    assert meta["capture_started_utc"] == t0.isoformat()
    # The diagnostic still measures the whole start_capture call.
    assert meta["start_capture_ms"] >= 100


def test_a_backend_without_t0_falls_back_to_the_end_of_start_capture():
    before = datetime.now(timezone.utc)
    meta = _run_scheduled_start(_Backend(None))
    started = datetime.fromisoformat(meta["capture_started_utc"])
    assert (started - before).total_seconds() >= 0.1


def test_sync_manager_reads_utc_with_its_monotonic_clock():
    sm = SyncManager()
    assert sm.start_utc is None
    before = datetime.now(timezone.utc)
    sm.start()
    assert before <= sm.start_utc <= datetime.now(timezone.utc)
    sm.reset()
    assert sm.start_utc is None
