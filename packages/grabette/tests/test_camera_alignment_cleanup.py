"""An alignment failure is a normal stop: every stream down, take kept as invalid."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from grabette.backend import rpi
from grabette.backend.rpi import RpiBackend
from grabette.hardware.camera import FrameAlignmentError


@pytest.mark.parametrize("alignment_fails", [False, True])
def test_stop_drains_other_streams_and_files_an_invalid_take_without_raising(
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
        frame_provenance = object()

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
    backend._stop_task = None
    backend._needs_reinit = False
    backend._speaker = None
    backend._angle = None
    backend._imu = None
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
    status = asyncio.run(backend.stop_capture())
    assert status.episode_id == episode.name
    assert status.valid is not alignment_fails
    assert sorted(events[:2]) == ["camera_closed", "oak_drained"]
    assert events[2:] == ["sync_reset", "finalizer_scheduled"]
    assert not backend._capturing and not backend._stopping and not backend._starting
    assert backend._needs_reinit and backend._finalizing
    assert backend._episode_dir is None
    assert len(scheduled) == 1
    callback, args = scheduled[0]
    saved_episode, timestamps, angles, metadata, urdf, provenance, imu = args
    assert imu is None
    assert provenance is backend._camera.frame_provenance
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


def test_finalizer_saves_frame_provenance_before_metadata(tmp_path):
    backend = object.__new__(RpiBackend)
    backend._speaker = Mock()
    backend._finalizing = True
    backend._reinit_hardware = Mock()
    order = []

    class Provenance:
        def save(self):
            order.append("provenance")
            assert not (tmp_path / "metadata.json").exists()
            return False

    metadata = {"video_timestamp_alignment": {"valid": True}}
    backend._finalize_and_reinit(
        tmp_path, [10.0], None, metadata, tmp_path / "missing-robot.urdf", Provenance())
    assert order == ["provenance"]
    assert json.loads((tmp_path / "metadata.json").read_text())["valid"] is False
    backend._speaker.play_error.assert_called_once_with()
    backend._speaker.play_saved.assert_not_called()


def test_shutdown_stops_an_invalid_take_and_releases_peripherals():
    backend = object.__new__(RpiBackend)
    backend._depth_camera_watch_task = None
    backend._capturing = True
    backend._cancel_oakd_keepalive = Mock()
    teleop = backend._teleop = Mock()
    oakd = backend._oakd = Mock()
    speaker = backend._speaker = Mock()

    async def stop_capture():
        backend._capturing = False
        return SimpleNamespace(valid=False)

    backend.stop_capture = stop_capture
    asyncio.run(backend.stop())
    teleop.shutdown.assert_called_once_with()
    oakd.shutdown.assert_called_once_with()
    speaker.close.assert_called_once_with()
    backend._cancel_oakd_keepalive.assert_called_once_with()
    assert backend._speaker is None
    assert backend._teleop is None
    assert not backend._running
