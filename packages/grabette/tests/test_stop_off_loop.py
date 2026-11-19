"""A long mux runs off the event loop; a stop arriving meanwhile joins the one in progress."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from grabette.backend import rpi
from grabette.backend.rpi import RpiBackend


def _backend(monkeypatch, tmp_path, camera):
    backend = object.__new__(RpiBackend)
    backend._capturing = True
    backend._starting = False
    backend._stopping = False
    backend._stop_task = None
    backend._needs_reinit = False
    backend._speaker = None
    backend._angle = None
    backend._sync = SimpleNamespace(get_timestamp_ms=lambda: 1000.0, reset=lambda: None)
    backend._camera = camera
    backend._oakd = None
    backend._oakd_auto_enabled = False
    backend._depth_camera = "oakd"
    backend._episode_dir = tmp_path / "take"
    backend._note_angle_output = Mock()
    backend._take_sync_metadata = Mock(return_value=None)
    backend._finalize_and_reinit = Mock()
    monkeypatch.setattr(rpi, "settings", SimpleNamespace(hand="left", device_id="test-left"))
    return backend


def test_the_loop_keeps_running_during_the_mux_and_a_second_stop_joins_it(monkeypatch, tmp_path):
    loop_ran = threading.Event()

    class Camera:
        frame_count = 2
        frame_timestamps = [10.0, 35.0]
        timestamp_alignment = {"valid": True, "errors": []}
        frame_provenance = None
        stops = 0

        def stop(self):
            self.stops += 1
            # Deadlocks (and fails) if the mux blocks the loop.
            assert loop_ran.wait(timeout=5)
            return self.frame_timestamps

    camera = Camera()
    backend = _backend(monkeypatch, tmp_path, camera)

    async def scenario():
        first = asyncio.ensure_future(backend.stop_capture())
        await asyncio.sleep(0.05)
        assert backend.is_stopping
        second = asyncio.ensure_future(backend.stop_capture())
        loop_ran.set()
        return await first, await second

    first, second = asyncio.run(scenario())

    assert first is second and first.episode_id == "take"
    assert camera.stops == 1
    assert backend._stop_task is None and not backend.is_capturing


def test_a_stop_after_the_last_one_ended_is_refused(monkeypatch, tmp_path):
    backend = _backend(monkeypatch, tmp_path, None)
    backend._capturing = False

    async def scenario():
        try:
            await backend.stop_capture()
        except RuntimeError as e:
            return str(e)

    assert asyncio.run(scenario()) == "Not capturing"
    assert backend._stop_task is None
