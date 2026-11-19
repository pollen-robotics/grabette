"""The fleet cannot delete an episode the device is still recording or writing."""
import asyncio
from types import SimpleNamespace

from grabette.app import main
from grabette.app.routers import tasks
from grabette.backend.rpi import RpiBackend
from grabette.task import TaskManager


def _delete(monkeypatch, tm, busy, eids):
    monkeypatch.setattr(main, "get_daemon_instance",
                        lambda: SimpleNamespace(backend=SimpleNamespace(busy_episode_id=busy)))
    monkeypatch.setattr(tasks, "get_task_manager", lambda: tm)
    return asyncio.run(main._dispatch_relay_command(
        {"id": "del", "type": "delete_episode", "args": {"episode_ids": eids}}))


def test_the_episode_being_saved_is_kept_and_the_others_go(monkeypatch, tmp_path):
    tm = TaskManager(data_dir=tmp_path)
    saving = tm.create_episode(episode_id="saving-take")
    tm.register_episode(saving)
    old = tm.create_episode(episode_id="old-take")
    tm.register_episode(old)

    result = _delete(monkeypatch, tm, "saving-take", ["saving-take", "old-take"])

    assert result["status"] == "error" and "still being recorded or saved" in result["message"]
    assert result["deleted"] == ["old-take"]
    assert tm.episode_dir(saving).is_dir()


def test_the_backend_names_the_take_until_its_files_are_written(tmp_path):
    backend = object.__new__(RpiBackend)
    backend._capturing, backend._episode_dir = True, tmp_path / "take"
    backend._finalizing, backend._finalizing_episode = False, None
    assert backend.busy_episode_id == "take"

    backend._capturing, backend._episode_dir = False, None
    backend._finalizing, backend._finalizing_episode = True, "take"
    assert backend.busy_episode_id == "take"

    backend._finalizing = False
    assert backend.busy_episode_id is None
