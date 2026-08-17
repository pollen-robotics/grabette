"""Regression tests for TaskManager episode-info reporting.

Runs against a temp data dir (no hardware). The headline case is the #79
regression: has_imu must be True for real OAK-D episodes (which write
oakd_imu.json), not only for legacy/mock episodes (imu_data.json).
"""
import json
import shutil
from pathlib import Path

import pytest

from grabette.config import settings
from grabette.task import UNASSIGNED_REPORT_LIMIT, UNASSIGNED_ID, TaskManager


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


# A solo/offline capture gets no membership from the fleet, but the device knows
# its own role (settings.hand is its slot). It stamps that itself, so the episode
# reaches the fleet fully described instead of role-less — role-less episodes are
# flagged incomplete and are SILENTLY skipped by dataset generation.

@pytest.fixture
def left_hand(monkeypatch):
    """Pin this device's identity so the expected members map is deterministic."""
    monkeypatch.setattr(settings, "hand", "left")
    monkeypatch.setattr(settings, "device_id", "dev-left-1")
    monkeypatch.setattr(settings, "device_name", "grabette-left")
    return {"left": {"device_id": "dev-left-1", "name": "grabette-left"}}


def test_solo_episode_stamps_its_own_role(tmp_path, left_hand):
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.create_task("Solo Task")
    tm.active_task_id = tid

    _button_press(tm, "ep_solo")

    task = tm._find_task(tid)
    assert task["episode_members"]["ep_solo"] == left_hand
    # First episode of a signature-less task → the signature is filled in, which
    # is what makes the task usable for dataset generation at all.
    assert task["device_signature"] == ["left"]


def test_solo_episode_in_unassigned_stamps_members_but_no_signature(tmp_path, left_hand):
    # Unassigned holds episodes of any provenance, so it must never carry a
    # signature — but the members are kept, ready for when the episode is filed.
    tm = TaskManager(data_dir=tmp_path)

    _button_press(tm, "ep_loose")

    unassigned = tm._find_task(UNASSIGNED_ID)
    assert unassigned["episode_members"]["ep_loose"] == left_hand
    assert "device_signature" not in unassigned


def test_solo_episode_never_overwrites_a_known_signature(tmp_path, left_hand):
    # The regression this guards: a solo press into a bimanual task must not
    # rewrite its signature to ["left"] — that would misdescribe every episode
    # already recorded there.
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.get_or_create_task("Bimanual Task")
    tm.create_episode(tid, episode_id="ep_pair",
                      members={"left": {"device_id": "dev-left-1", "name": "grabette-left"},
                               "right": {"device_id": "dev-right-1", "name": "grabette-right"}},
                      signature=["left", "right"])
    tm.register_episode("ep_pair")
    tm.active_task_id = tid

    _button_press(tm, "ep_half_rig")

    task = tm._find_task(tid)
    assert task["device_signature"] == ["left", "right"]
    # The half-rig episode still says truthfully what recorded it.
    assert task["episode_members"]["ep_half_rig"] == left_hand


def test_fleet_membership_wins_over_self_reporting(tmp_path, left_hand):
    # When the fleet describes the episode, its view is authoritative — the
    # self-stamp must not narrow a group episode down to this one device.
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.get_or_create_task("Group Task")
    members = {"left": {"device_id": "dev-left-1", "name": "grabette-left"},
               "casquette": {"device_id": "dev-cap-1", "name": "casquette"}}
    tm.create_episode(tid, episode_id="ep_grp", members=members,
                      signature=["left", "casquette"])
    tm.register_episode("ep_grp")

    task = tm._find_task(tid)
    assert task["episode_members"]["ep_grp"] == members
    assert task["device_signature"] == ["left", "casquette"]


def test_solo_episode_is_reported_with_its_role(tmp_path, left_hand):
    # End to end: report_tasks is what the fleet aggregates, so the role has to
    # survive into the report for the episode to count as complete there.
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.create_task("Reported Solo")
    tm.active_task_id = tid

    _button_press(tm, "ep_rep")

    entry = next(t for t in tm.report_tasks() if t["name"] == "Reported Solo")
    assert entry["device_signature"] == ["left"]
    assert entry["groups"] == [{"members": left_hand, "episode_ids": ["ep_rep"]}]


# Membership lives on the dict of the task that HOLDS the episode, so every path
# that refiles an episode has to carry it across. Otherwise the role stamped at
# record time is silently dropped exactly when a loose recording gets filed —
# the one moment it matters.

PAIR = {"left": {"device_id": "dev-left-1", "name": "grabette-left"},
        "right": {"device_id": "dev-right-1", "name": "grabette-right"}}


def _fleet_episode(tm, task_id, episode_id, members=PAIR, signature=("left", "right")):
    """An episode recorded through the fleet: it arrives fully described."""
    tm.create_episode(task_id, episode_id=episode_id, members=members,
                      signature=list(signature))
    tm.register_episode(episode_id)


def test_move_carries_members_and_seeds_the_signature(tmp_path, left_hand):
    tm = TaskManager(data_dir=tmp_path)
    _button_press(tm, "ep_loose")  # lands in Unassigned, self-stamped
    tid = tm.create_task("Filed Task")

    tm.move_episodes(["ep_loose"], tid)

    task = tm._find_task(tid)
    assert task["episode_members"]["ep_loose"] == left_hand
    # A fresh task with no signature would expose no roles at all, and dataset
    # generation refuses such a task — so filing has to seed it.
    assert task["device_signature"] == ["left"]
    # And nothing is left stranded on the source.
    assert "ep_loose" not in tm._find_task(UNASSIGNED_ID).get("episode_members", {})


def test_move_never_replaces_an_existing_signature(tmp_path, left_hand):
    # Filing a mono episode into a bimanual task is the half-rig case. The device
    # stays mechanical about it — refusing belongs to the fleet, which knows both
    # sides — but it must not rewrite the task's signature to ["left"].
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.get_or_create_task("Bimanual Task")
    _fleet_episode(tm, tid, "ep_pair")
    _button_press(tm, "ep_loose")

    tm.move_episodes(["ep_loose"], tid)

    task = tm._find_task(tid)
    assert task["device_signature"] == ["left", "right"]
    assert task["episode_members"]["ep_loose"] == left_hand
    assert task["episode_members"]["ep_pair"] == PAIR


def test_move_legacy_episode_without_members_invents_nothing(tmp_path):
    # Migrated episodes are registered as bare ids (see _migrate_legacy), with no
    # membership: moving one must neither crash nor fabricate a signature.
    _make_episode(tmp_path, "ep_old", ["oakd_imu.json"])
    tm = TaskManager(data_dir=tmp_path)
    tm._find_task(UNASSIGNED_ID)["episode_ids"].append("ep_old")
    tid = tm.create_task("Target")

    tm.move_episodes(["ep_old"], tid)

    task = tm._find_task(tid)
    assert task["episode_ids"] == ["ep_old"]
    assert task.get("episode_members", {}) == {}
    assert "device_signature" not in task


def test_move_into_the_holding_task_preserves_members(tmp_path, left_hand):
    # The trap: the source lookup finds the TARGET as holder, so a naive
    # implementation strips its members and then skips the re-add.
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.create_task("Task")
    tm.active_task_id = tid
    _button_press(tm, "ep_a")

    tm.move_episodes(["ep_a"], tid)

    task = tm._find_task(tid)
    assert task["episode_ids"] == ["ep_a"]
    assert task["episode_members"]["ep_a"] == left_hand


def test_move_with_divergent_roles_leaves_the_signature_unset(tmp_path, left_hand):
    # Unassigned can legitimately hold episodes of different shapes: a bimanual
    # one that fell back when its task was deleted, plus a mono button press.
    # Their union ["left","right"] would describe neither, so nothing is claimed.
    tm = TaskManager(data_dir=tmp_path)
    doomed = tm.get_or_create_task("Doomed")
    _fleet_episode(tm, doomed, "ep_pair")
    tm.delete_task(doomed)
    _button_press(tm, "ep_solo")

    tid = tm.create_task("Mixed")
    tm.move_episodes(["ep_pair", "ep_solo"], tid)

    task = tm._find_task(tid)
    assert "device_signature" not in task
    assert task["episode_members"]["ep_pair"] == PAIR
    assert task["episode_members"]["ep_solo"] == left_hand


def test_delete_task_carries_members_to_unassigned(tmp_path, left_hand):
    # delete_task drops the task dict, so its episode_members go with it unless
    # they are carried over — an unrecoverable loss, not just a stranded entry.
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.create_task("Task A")
    tm.active_task_id = tid
    _button_press(tm, "ep_a")

    tm.delete_task(tid)

    unassigned = tm._find_task(UNASSIGNED_ID)
    assert "ep_a" in unassigned["episode_ids"]
    assert unassigned["episode_members"]["ep_a"] == left_hand


def test_moving_into_unassigned_never_sets_a_signature(tmp_path, left_hand):
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.create_task("Task")
    tm.active_task_id = tid
    _button_press(tm, "ep_a")

    tm.move_episodes(["ep_a"], UNASSIGNED_ID)

    unassigned = tm._find_task(UNASSIGNED_ID)
    assert unassigned["episode_members"]["ep_a"] == left_hand
    assert "device_signature" not in unassigned


# The inbox travels on its own channel, never as a task: tasks get merged across
# devices by name on the fleet, which would make every device's stray takes
# indistinguishable — and would let them reach dataset generation.

def test_report_unassigned_describes_loose_episodes(tmp_path, left_hand):
    tm = TaskManager(data_dir=tmp_path)
    _button_press(tm, "ep_loose")
    (tmp_path / "episodes" / "ep_loose" / "raw_video.mp4").write_text("x")
    (tmp_path / "episodes" / "ep_loose" / "metadata.json").write_text(
        json.dumps({"duration_seconds": 12.5}))

    report = tm.report_unassigned()

    assert report["total"] == 1
    assert report["episodes"] == [{
        "episode_id": "ep_loose",
        "members": left_hand,          # who recorded it — this device, self-stamped
        "duration_seconds": 12.5,      # enough to tell a real take from a misfire
        "has_video": True,
    }]


def test_report_unassigned_is_empty_when_everything_is_filed(tmp_path, left_hand):
    tm = TaskManager(data_dir=tmp_path)
    tid = tm.create_task("Filed")
    tm.active_task_id = tid
    _button_press(tm, "ep_filed")

    assert tm.report_unassigned() == {"total": 0, "episodes": []}


def test_loose_episodes_stay_out_of_the_task_report(tmp_path, left_hand):
    # The two channels must not overlap: report_tasks still skips Unassigned, so
    # a loose episode can never be selected for a dataset.
    tm = TaskManager(data_dir=tmp_path)
    _button_press(tm, "ep_loose")

    assert tm.report_tasks() == []
    assert tm.report_unassigned()["total"] == 1


def test_report_unassigned_skips_episodes_without_data(tmp_path, left_hand):
    # A registry entry whose directory is gone can't be triaged — offering it
    # would only produce actions that fail.
    tm = TaskManager(data_dir=tmp_path)
    _button_press(tm, "ep_gone")
    shutil.rmtree(tmp_path / "episodes" / "ep_gone")

    assert tm.report_unassigned() == {"total": 0, "episodes": []}


def test_report_unassigned_caps_the_payload_but_not_the_count(tmp_path, left_hand):
    # Truncation must stay visible: `total` is the honest count, and the entries
    # kept are the most recent ones an operator would actually triage.
    tm = TaskManager(data_dir=tmp_path)
    n = UNASSIGNED_REPORT_LIMIT + 5
    for i in range(n):
        _button_press(tm, f"ep_{i:04d}")

    report = tm.report_unassigned()

    assert report["total"] == n
    assert len(report["episodes"]) == UNASSIGNED_REPORT_LIMIT
    assert report["episodes"][-1]["episode_id"] == f"ep_{n - 1:04d}"
