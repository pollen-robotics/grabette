"""An alignment failure must not leave other streams or capture state live."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from grabette.backend import rpi
from grabette.backend.rpi import RpiBackend
from grabette.hardware.camera import FrameAlignmentError


@pytest.mark.parametrize("alignment_fails", [False, True])
def test_stop_drains_other_streams_and_defers_diagnostics_before_reporting_error(
    monkeypatch, tmp_path, alignment_fails,
):
    backend = object.__new__(RpiBackend)
    events, scheduled = [], []
    failure = FrameAlignmentError("injected unmapped output PTS")

    class Sync:
        reset_done = False

        def get_timestamp_ms(self):
            return 1000.0

        def reset(self):
            self.reset_done = True
            events.append("sync_reset")

    class Camera:
        frame_count = 2
        frame_timestamps = [10.0, 35.0]
        timestamp_alignment = {"valid": not alignment_fails, "errors": []}

        def stop(self):
            events.append("camera_closed")
            if alignment_fails:
                raise failure
            return self.frame_timestamps

    class Oak:
        is_recording = True
        is_initialized = True

        def stop_recording(self):
            assert backend._capturing and backend._stopping and backend._needs_reinit
            self.is_recording = False
            events.append("oak_drained")
            return {"imu_samples": 60}

        def camera_info(self):
            return {"name": "test stereo camera"}

    class Loop:
        def run_in_executor(self, executor, function):
            async def drain():
                await asyncio.sleep(0)
                return function()

            return asyncio.create_task(drain())

        def call_soon(self, callback, *args):
            assert not backend._capturing and not backend._stopping and backend._sync.reset_done
            scheduled.append((callback, args))
            events.append("finalizer_scheduled")

    backend._capturing = True
    backend._starting = True
    backend._stopping = False
    backend._needs_reinit = False
    backend._speaker = None
    backend._angle = None
    backend._sync = Sync()
    backend._camera = Camera()
    backend._oakd = Oak()
    backend._oakd_auto_enabled = True
    backend._depth_camera = "oakd"
    episode = tmp_path / "test-episode"
    backend._episode_dir = episode
    backend._note_angle_output = Mock()
    backend._schedule_oakd_keepalive = Mock()
    backend._take_sync_metadata = Mock(return_value={"group_id": "test-group"})
    backend._finalize_and_reinit = Mock()
    monkeypatch.setattr(rpi, "settings", SimpleNamespace(hand="left", device_id="test-left"))
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: Loop())
    if alignment_fails:
        with pytest.raises(FrameAlignmentError) as caught:
            asyncio.run(backend.stop_capture())
        assert caught.value is failure
        assert failure.episode_id == episode.name
    else:
        status = asyncio.run(backend.stop_capture())
        assert status.episode_id == episode.name
    assert events == ["camera_closed", "oak_drained", "sync_reset", "finalizer_scheduled"]
    assert not backend._capturing and not backend._stopping and not backend._starting
    assert backend._needs_reinit and backend._finalizing
    assert backend._episode_dir is None
    assert len(scheduled) == 1
    callback, args = scheduled[0]
    saved_episode, timestamps, angles, metadata, urdf = args
    assert callback is backend._finalize_and_reinit
    assert saved_episode == episode
    assert timestamps == [10.0, 35.0]
    assert angles is None
    assert metadata["video_timestamp_alignment"]["valid"] is not alignment_fails
    if alignment_fails:
        assert metadata["valid"] is False
    assert metadata["imu_sample_count"] == 60
    assert metadata["sync"] == {"group_id": "test-group"}
    assert urdf.name == "robot.urdf"
    backend._schedule_oakd_keepalive.assert_called_once_with()
    backend._finalize_and_reinit.assert_not_called()


def test_invalid_alignment_is_persisted_without_the_successful_save_cue(monkeypatch, tmp_path):
    backend = object.__new__(RpiBackend)
    backend._speaker = Mock()
    backend._finalizing = True
    backend._reinit_hardware = Mock()
    monkeypatch.setattr(rpi, "_CAMERA_INTRINSICS_SRC", tmp_path / "missing-intrinsics.json")
    metadata = {"valid": False, "video_timestamp_alignment": {"valid": False}}
    backend._finalize_and_reinit(
        tmp_path, [10.0, 35.0], None, metadata, tmp_path / "missing-robot.urdf")
    assert json.loads((tmp_path / "metadata.json").read_text())["valid"] is False
    assert json.loads((tmp_path / "frame_timestamps.json").read_text()) == [10.0, 35.0]
    backend._speaker.play_error.assert_called_once_with()
    backend._speaker.play_saved.assert_not_called()
    assert not backend._finalizing
    backend._reinit_hardware.assert_called_once_with()


def test_shutdown_releases_peripherals_before_reraising_alignment_error():
    backend = object.__new__(RpiBackend)
    failure = FrameAlignmentError("invalid stopped take", episode_id="test-episode")
    backend._depth_camera_watch_task = None
    backend._capturing = True
    backend._cancel_oakd_keepalive = Mock()
    teleop = backend._teleop = Mock()
    oakd = backend._oakd = Mock()
    speaker = backend._speaker = Mock()

    async def stop_capture():
        backend._capturing = False
        raise failure

    backend.stop_capture = stop_capture
    with pytest.raises(FrameAlignmentError) as caught:
        asyncio.run(backend.stop())
    assert caught.value is failure
    teleop.shutdown.assert_called_once_with()
    oakd.shutdown.assert_called_once_with()
    speaker.close.assert_called_once_with()
    backend._cancel_oakd_keepalive.assert_called_once_with()
    assert backend._speaker is None
    assert backend._teleop is None
    assert not backend._running
