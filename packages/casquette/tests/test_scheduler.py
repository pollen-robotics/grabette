"""EpisodeScheduler state-machine tests.

Covers the IDLE → SCHEDULED → STARTING → RECORDING transitions and the
sync-metadata fold that lets a multi-device episode be reconstructed from
a single device's metadata.json.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from casquette.scheduler import CaptureState

# A T0 far enough out that the scheduled task is still waiting when we
# assert on it (no race with the sleep firing).
FAR_FUTURE_S = 30
# A T0 close enough that the test doesn't wait long, but comfortably above
# scheduling jitter so the fire is reliable.
NEAR_FUTURE_S = 0.2


def _utc_in(seconds: float) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


async def test_immediate_start_records(scheduler, sm):
    episode_id = await scheduler.start()
    assert scheduler.state == CaptureState.RECORDING
    assert sm.episode_dir(episode_id).exists()


async def test_start_when_not_idle_raises(scheduler):
    await scheduler.start()
    with pytest.raises(RuntimeError):
        await scheduler.start()


async def test_past_start_time_raises_valueerror(scheduler):
    with pytest.raises(ValueError):
        await scheduler.start(start_at_utc=_utc_in(-5))


async def test_scheduled_start_enters_scheduled(scheduler):
    target = _utc_in(FAR_FUTURE_S)
    episode_id = await scheduler.start(start_at_utc=target)
    assert scheduler.state == CaptureState.SCHEDULED
    assert scheduler.scheduled_episode_id == episode_id
    assert scheduler.scheduled_at_utc == target


async def test_stop_while_scheduled_cancels_and_deletes(scheduler, sm):
    episode_id = await scheduler.start(start_at_utc=_utc_in(FAR_FUTURE_S))
    result = await scheduler.stop()
    assert result is None  # nothing was recorded
    assert scheduler.state == CaptureState.IDLE
    assert not sm.episode_dir(episode_id).exists()  # dir cleaned up


async def test_stop_while_idle_raises(scheduler):
    with pytest.raises(RuntimeError):
        await scheduler.stop()


async def test_stop_while_recording_returns_status(scheduler):
    await scheduler.start()
    status = await scheduler.stop()
    assert status is not None
    assert scheduler.state == CaptureState.IDLE


async def test_scheduled_fires_and_writes_sync_metadata(scheduler, sm):
    peers = [
        {"device_id": "casquette-1", "url": None},
        {"device_id": "grabette-2", "url": "http://grabette2:8000"},
    ]
    target = _utc_in(NEAR_FUTURE_S)
    episode_id = await scheduler.start(start_at_utc=target, peers=peers)

    # Wait past T0 for the scheduled task to run start_capture.
    await asyncio.sleep(NEAR_FUTURE_S + 0.3)
    assert scheduler.state == CaptureState.RECORDING

    # Stopping folds the sync metadata into metadata.json.
    await scheduler.stop()
    meta = json.loads((sm.episode_dir(episode_id) / "metadata.json").read_text())
    assert "sync" in meta
    assert meta["sync"]["scheduled_start_utc"] == target.isoformat()
    assert meta["sync"]["peers"] == peers
    assert "sleep_end_skew_ms" in meta["sync"]


async def test_local_only_start_has_no_sync_metadata(scheduler, sm):
    # An immediate, peer-less start must not inherit sync metadata.
    episode_id = await scheduler.start()
    await scheduler.stop()
    meta = json.loads((sm.episode_dir(episode_id) / "metadata.json").read_text())
    assert "sync" not in meta
