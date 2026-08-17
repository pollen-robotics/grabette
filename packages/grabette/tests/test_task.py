"""Regression tests for TaskManager episode-info reporting.

Runs against a temp data dir (no hardware). The headline case is the #79
regression: has_imu must be True for real OAK-D episodes (which write
oakd_imu.json), not only for legacy/mock episodes (imu_data.json).
"""
import json
from pathlib import Path

from grabette.task import UNASSIGNED_ID, TaskManager


def _make_episode(data_dir: Path, episode_id: str, files, meta=None) -> Path:
    """Create <data_dir>/episodes/<id>/ with the given files + a metadata.json."""
    ep = data_dir / "episodes" / episode_id
    ep.mkdir(parents=True)
    for name in files:
        (ep / name).write_text("{}")  # content is irrelevant to has_* / counts
    (ep / "metadata.json").write_text(json.dumps(meta or {}))
    return ep


def test_has_imu_oakd_episode(tmp_path):
    # #79: a real OAK-D episode writes oakd_imu.json — must report has_imu.
    _make_episode(tmp_path, "ep_oak", ["oakd_imu.json", "raw_video.mp4"])
    info = TaskManager(data_dir=tmp_path)._get_episode_info("ep_oak")
    assert info.has_imu is True
    assert info.has_video is True


def test_has_imu_legacy_episode(tmp_path):
    # Legacy/mock episodes write imu_data.json — must still report has_imu.
    _make_episode(tmp_path, "ep_legacy", ["imu_data.json", "raw_video.mp4"])
    info = TaskManager(data_dir=tmp_path)._get_episode_info("ep_legacy")
    assert info.has_imu is True


def test_has_imu_absent(tmp_path):
    _make_episode(tmp_path, "ep_none", ["raw_video.mp4"])
    info = TaskManager(data_dir=tmp_path)._get_episode_info("ep_none")
    assert info.has_imu is False


def test_has_video_absent(tmp_path):
    _make_episode(tmp_path, "ep_novideo", ["oakd_imu.json"])
    info = TaskManager(data_dir=tmp_path)._get_episode_info("ep_novideo")
    assert info.has_video is False


def test_imu_sample_count_from_metadata(tmp_path):
    # imu_sample_count is read from metadata (oakd.imu_samples), not the file.
    _make_episode(tmp_path, "ep_cnt", ["oakd_imu.json"], meta={"oakd": {"imu_samples": 2115}})
    info = TaskManager(data_dir=tmp_path)._get_episode_info("ep_cnt")
    assert info.imu_sample_count == 2115


# The two delete paths are separate methods here rather than one call with a
# delete_episodes flag: delete_task() is the non-destructive path (episodes fall
# back to Unassigned), while delete_task_by_name() is the fleet's destructive
# path (the recordings go too). See #98.

def test_delete_task_keeps_episodes(tmp_path):
    # #98: delete_task() preserves the episodes, reassigning them to Unassigned.
    _make_episode(tmp_path, "ep_a", ["oakd_imu.json"])
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.create_task("Task A")
    tm.move_episodes(["ep_a"], tid)

    tm.delete_task(tid)

    assert (tmp_path / "episodes" / "ep_a").exists()
    assert "ep_a" in tm.get_task_detail(UNASSIGNED_ID).episode_ids


def test_delete_task_by_name_purges_episodes(tmp_path):
    # #98: delete_task_by_name() removes the episode dirs from disk and does not
    # leak them into Unassigned.
    _make_episode(tmp_path, "ep_b", ["oakd_imu.json"])
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.create_task("Task B")
    tm.move_episodes(["ep_b"], tid)

    assert tm.delete_task_by_name("Task B") is True

    assert not (tmp_path / "episodes" / "ep_b").exists()
    assert "ep_b" not in tm.get_task_detail(UNASSIGNED_ID).episode_ids


def test_delete_task_by_name_absent_is_noop(tmp_path):
    # Fleet deletes are dispatched to every device and must stay idempotent:
    # a device that never had the task still reports success-as-absent.
    tm = TaskManager(data_dir=tmp_path)
    assert tm.delete_task_by_name("Never Existed") is False


# A recording started outside any session must land in Unassigned — visibly
# untriaged — never inside the task that was last recorded to. Two paths used to
# leak: the local session lock kept its task selected after stop_session(), and
# register_episode() moved active_task_id onto every task it filed into (so a
# fleet-driven session, which never calls start/stop_session on the device, left
# it pointing at the group's task once closed).

def _button_press(tm, episode_id):
    """Simulate a physical-button press outside a session: the listener resolves
    the target from active_task_id, then files the episode on stop."""
    tm.create_episode(tm.active_task_id, episode_id=episode_id)
    tm.register_episode(episode_id)


def test_button_press_after_local_session_goes_to_unassigned(tmp_path):
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.create_task("Task A")
    tm.start_session(tid)
    tm.create_episode(tid, episode_id="ep_in_session")
    tm.register_episode("ep_in_session")
    tm.stop_session()

    _button_press(tm, "ep_after_session")

    assert tm.active_task_id == UNASSIGNED_ID
    assert "ep_after_session" in tm.get_task_detail(UNASSIGNED_ID).episode_ids
    assert tm.get_task_detail(tid).episode_ids == ["ep_in_session"]


def test_button_press_after_fleet_session_goes_to_unassigned(tmp_path):
    # Fleet-driven session: episodes are filed against the group's task without
    # the device's session lock ever being used, and closing the session on the
    # fleet sends the device nothing. The next press must still be Unassigned.
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.get_or_create_task("Group Task")
    tm.create_episode(tid, episode_id="ep_group")
    tm.register_episode("ep_group")

    _button_press(tm, "ep_after_group")

    assert tm.active_task_id == UNASSIGNED_ID
    assert "ep_after_group" in tm.get_task_detail(UNASSIGNED_ID).episode_ids
    assert tm.get_task_detail(tid).episode_ids == ["ep_group"]


def test_explicit_local_selection_survives_a_recording(tmp_path):
    # The flip side: an explicitly selected task (local UI) stays selected, so
    # hands-free recording of several episodes into it keeps working.
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.create_task("Task B")
    tm.active_task_id = tid

    _button_press(tm, "ep_b1")
    _button_press(tm, "ep_b2")

    assert tm.active_task_id == tid
    assert tm.get_task_detail(tid).episode_ids == ["ep_b1", "ep_b2"]
