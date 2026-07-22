"""Regression tests for SessionManager episode-info reporting.

Runs against a temp data dir (no hardware). The headline case is the #79
regression: has_imu must be True for real OAK-D episodes (which write
oakd_imu.json), not only for legacy/mock episodes (imu_data.json).
"""
import json
from pathlib import Path

from grabette.session import SessionManager


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
