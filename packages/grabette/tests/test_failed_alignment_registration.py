"""A stopped invalid take is a normal stop: filed once, no error, no second cue."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from grabette import button_listener, capture_scheduler, fleet_sync
from grabette.capture_scheduler import CaptureScheduler, stop_capture_and_register
from grabette.models import CaptureStatus
from grabette.task import TaskManager


@pytest.mark.parametrize("entrypoint", ["rest", "button", "relay", "scheduled"])
def test_each_stop_entrypoint_files_an_invalid_take_without_failing(
    monkeypatch, tmp_path, entrypoint,
):
    # REST/main imports create a module-level TaskManager. Keep that default
    # instance in the test tree too, without replacing the real class/methods.
    original_init = TaskManager.__init__

    def isolated_init(self, data_dir=None):
        original_init(self, data_dir=data_dir if data_dir is not None
                      else tmp_path / "import-task-state")

    monkeypatch.setattr(TaskManager, "__init__", isolated_init)
    tm = TaskManager(data_dir=tmp_path)
    episode = tm.create_episode(episode_id="failed-take")
    (tm.episode_dir(episode) / "metadata.json").write_text(json.dumps({"valid": False}))
    register = Mock(wraps=tm.register_episode)
    monkeypatch.setattr(tm, "register_episode", register)
    backend = SimpleNamespace(is_capturing=True)
    status = CaptureStatus(episode_id=episode, valid=False)

    async def stop_capture():
        backend.is_capturing = False
        return status

    backend.stop_capture = stop_capture
    scheduler = CaptureScheduler()
    scheduler.cancel_or_wait = AsyncMock(return_value="ran")
    monkeypatch.setattr(capture_scheduler, "get_capture_scheduler", lambda: scheduler)
    monkeypatch.setattr(fleet_sync, "notify_group_stop", AsyncMock())
    # The finalizer plays the only error cue.
    cue = Mock()
    monkeypatch.setattr(capture_scheduler, "cue_error", cue)
    monkeypatch.setattr(button_listener, "cue_error", cue)

    async def invoke():
        if entrypoint == "rest":
            from grabette.app.routers import tasks
            return await tasks.stop_capture(backend=backend, tm=tm)
        if entrypoint == "button":
            from grabette.button_listener import ButtonListener
            return await ButtonListener(backend, tm)._stop_capture_coro()
        if entrypoint == "relay":
            from grabette.app import main
            from grabette.app.routers import tasks
            from grabette.daemon import DaemonState
            monkeypatch.setattr(main, "get_daemon_instance", lambda: SimpleNamespace(
                backend=backend, state=DaemonState.RUNNING))
            monkeypatch.setattr(tasks, "get_task_manager", lambda: tm)
            return await main._dispatch_relay_command({"id": "test-stop", "type": "stop_capture"})
        return await scheduler._wait_and_stop(backend, tm, datetime.now(timezone.utc))

    result = asyncio.run(invoke())
    if entrypoint == "rest":
        assert result is status and result.valid is False
    elif entrypoint == "relay":
        assert result["status"] == "ok" and result["result"]["valid"] is False
    cue.assert_not_called()
    register.assert_called_once_with(episode, valid=False)
    tm.create_episode(episode_id="next-take")
    assert episode in {eid for task in tm.list_tasks() for eid in task.episode_ids}
    assert tm.episode_dir(episode).is_dir()


def test_a_failed_stop_registers_nothing():
    failure = RuntimeError("generic mux failure")
    backend = SimpleNamespace(is_capturing=False, stop_capture=AsyncMock(side_effect=failure))
    tm = Mock()
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(stop_capture_and_register(backend, tm))
    assert caught.value is failure
    tm.register_episode.assert_not_called()


def test_successful_stop_keeps_its_existing_registration_contract():
    result = SimpleNamespace(episode_id="saved-take", valid=True)
    backend = SimpleNamespace(stop_capture=AsyncMock(return_value=result))
    tm = Mock()
    assert asyncio.run(stop_capture_and_register(backend, tm)) is result
    tm.register_episode.assert_called_once_with("saved-take", valid=True)
