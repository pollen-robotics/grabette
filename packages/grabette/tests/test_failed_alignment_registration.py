"""A stopped invalid take stays discoverable after the next take is created."""
import asyncio
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from grabette import capture_scheduler, fleet_sync
from grabette.capture_scheduler import CaptureScheduler, stop_capture_and_register
from grabette.hardware.camera import FrameAlignmentError
from grabette.task import TaskManager


@pytest.mark.parametrize("entrypoint", ["rest", "button", "relay", "scheduled"])
def test_each_stop_entrypoint_registers_the_exact_failed_take_once(
    monkeypatch, tmp_path, caplog, entrypoint,
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
    original_register = tm.register_episode
    register = Mock(wraps=original_register)
    monkeypatch.setattr(tm, "register_episode", register)
    failure = FrameAlignmentError("unmapped encoded picture", episode_id=episode)
    backend = SimpleNamespace(is_capturing=True)

    async def stop_capture():
        backend.is_capturing = False
        raise failure

    backend.stop_capture = stop_capture
    scheduler = CaptureScheduler()
    scheduler.cancel_or_wait = AsyncMock(return_value="ran")
    monkeypatch.setattr(capture_scheduler, "get_capture_scheduler", lambda: scheduler)
    monkeypatch.setattr(fleet_sync, "notify_group_stop", AsyncMock())
    # The background scheduled-stop API already reports failures through its
    # logger and audible cue. The other three entrypoints propagate the error.
    cue = Mock()
    monkeypatch.setattr(capture_scheduler, "cue_error", cue)

    async def invoke():
        if entrypoint == "rest":
            from grabette.app.routers import tasks
            await tasks.stop_capture(backend=backend, tm=tm)
        elif entrypoint == "button":
            from grabette.button_listener import ButtonListener
            await ButtonListener(backend, tm)._stop_capture_coro()
        elif entrypoint == "relay":
            from grabette.app import main
            from grabette.app.routers import tasks
            from grabette.daemon import DaemonState
            monkeypatch.setattr(main, "get_daemon_instance", lambda: SimpleNamespace(
                backend=backend, state=DaemonState.RUNNING))
            monkeypatch.setattr(tasks, "get_task_manager", lambda: tm)
            await main._dispatch_relay_command({"id": "test-stop", "type": "stop_capture"})
        else:
            await scheduler._wait_and_stop(backend, tm, datetime.now(timezone.utc))

    if entrypoint == "scheduled":
        with caplog.at_level(logging.ERROR):
            asyncio.run(invoke())
        assert any(record.exc_info and record.exc_info[1] is failure for record in caplog.records)
        cue.assert_called_once_with()
    else:
        with pytest.raises(FrameAlignmentError) as caught:
            asyncio.run(invoke())
        assert caught.value is failure
    register.assert_called_once_with(episode)
    # Reproduces the original orphaning boundary: create_episode replaces the
    # one pending slot. The failed take must already have its own registry row.
    tm.create_episode(episode_id="next-take")
    assert episode in {eid for task in tm.list_tasks() for eid in task.episode_ids}
    assert tm.episode_dir(episode).is_dir()


@pytest.mark.parametrize("failure,still_capturing", [
    (RuntimeError("generic mux failure"), False),
    (FrameAlignmentError("no completed episode identity"), False),
    (FrameAlignmentError("stop still in progress", episode_id="failed-take"), True),
])
def test_other_stop_failures_do_not_register_an_unfinished_or_unidentified_take(failure, still_capturing):
    backend = SimpleNamespace(
        is_capturing=still_capturing, stop_capture=AsyncMock(side_effect=failure))
    tm = Mock()
    with pytest.raises(type(failure)) as caught:
        asyncio.run(stop_capture_and_register(backend, tm))
    assert caught.value is failure
    tm.register_episode.assert_not_called()


def test_successful_stop_keeps_its_existing_registration_contract():
    result = SimpleNamespace(episode_id="saved-take")
    backend = SimpleNamespace(stop_capture=AsyncMock(return_value=result))
    tm = Mock()
    assert asyncio.run(stop_capture_and_register(backend, tm)) is result
    tm.register_episode.assert_called_once_with("saved-take")
