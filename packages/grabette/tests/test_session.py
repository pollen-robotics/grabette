"""Regression tests for SessionManager episode-info reporting.

Runs against a temp data dir (no hardware). The headline case is the #79
regression: has_imu must be True for real OAK-D episodes (which write
oakd_imu.json), not only for legacy/mock episodes (imu_data.json).
"""
import json
from pathlib import Path

from grabette.session import UNASSIGNED_ID, SessionManager


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
    info = SessionManager(data_dir=tmp_path)._get_episode_info("ep_oak")
    assert info.has_imu is True
    assert info.has_video is True


def test_has_imu_legacy_episode(tmp_path):
    # Legacy/mock episodes write imu_data.json — must still report has_imu.
    _make_episode(tmp_path, "ep_legacy", ["imu_data.json", "raw_video.mp4"])
    info = SessionManager(data_dir=tmp_path)._get_episode_info("ep_legacy")
    assert info.has_imu is True


def test_has_imu_absent(tmp_path):
    _make_episode(tmp_path, "ep_none", ["raw_video.mp4"])
    info = SessionManager(data_dir=tmp_path)._get_episode_info("ep_none")
    assert info.has_imu is False


def test_has_video_absent(tmp_path):
    _make_episode(tmp_path, "ep_novideo", ["oakd_imu.json"])
    info = SessionManager(data_dir=tmp_path)._get_episode_info("ep_novideo")
    assert info.has_video is False


def test_imu_sample_count_from_metadata(tmp_path):
    # imu_sample_count is read from metadata (oakd.imu_samples), not the file.
    _make_episode(tmp_path, "ep_cnt", ["oakd_imu.json"], meta={"oakd": {"imu_samples": 2115}})
    info = SessionManager(data_dir=tmp_path)._get_episode_info("ep_cnt")
    assert info.imu_sample_count == 2115


def test_delete_session_keeps_episodes(tmp_path):
    # #98: deleting a task with delete_episodes=False preserves the episodes,
    # reassigning them to Unassigned (the default, non-destructive path).
    _make_episode(tmp_path, "ep_a", ["oakd_imu.json"])
    sm = SessionManager(data_dir=tmp_path)
    sid = sm.create_session("Task A")
    sm.move_episodes(["ep_a"], sid)

    sm.delete_session(sid, delete_episodes=False)

    assert (tmp_path / "episodes" / "ep_a").exists()
    assert "ep_a" in sm.get_session_detail(UNASSIGNED_ID).episode_ids


def test_delete_session_purges_episodes(tmp_path):
    # #98: deleting a task with delete_episodes=True removes the episode dirs
    # from disk and does not leak them into Unassigned.
    _make_episode(tmp_path, "ep_b", ["oakd_imu.json"])
    sm = SessionManager(data_dir=tmp_path)
    sid = sm.create_session("Task B")
    sm.move_episodes(["ep_b"], sid)

    sm.delete_session(sid, delete_episodes=True)

    assert not (tmp_path / "episodes" / "ep_b").exists()
    assert "ep_b" not in sm.get_session_detail(UNASSIGNED_ID).episode_ids
