"""The on-device readers must handle both episode layouts.

Captures write `dcam_*` now. Episodes already sitting in ~/grabette-data on
deployed devices use `oakd_*`, and the dashboard and replay engine read them
directly — so if the fallback breaks, an operator's existing episodes start
reporting "no IMU" and replaying without accelerometer data.

Mirrors packages/grabette-postprocess/tests/test_episode_files.py; the two
resolvers are deliberate duplicates because the packages are independent
distributions, so both need their own coverage.
"""
import json

import pytest

from grabette.hardware.episode_files import (
    CANONICAL_META_KEY,
    DCAM_IMU,
    DCAM_MASK,
    LEGACY_META_KEY,
    LEGACY_NAMES,
    legacy_name,
    metadata_stats,
    resolve,
)


def test_every_canonical_name_is_dcam_prefixed():
    for canonical in LEGACY_NAMES:
        assert canonical.startswith("dcam_"), canonical


def test_the_mask_is_the_one_irregular_rename():
    assert legacy_name(DCAM_MASK) == "oak_mask.png"


def test_resolve_prefers_canonical_then_falls_back(tmp_path):
    assert resolve(tmp_path, DCAM_IMU).name == DCAM_IMU          # neither present
    (tmp_path / "oakd_imu.json").write_text("{}")
    assert resolve(tmp_path, DCAM_IMU).name == "oakd_imu.json"   # legacy only
    (tmp_path / DCAM_IMU).write_text("{}")
    assert resolve(tmp_path, DCAM_IMU).name == DCAM_IMU          # canonical wins


@pytest.mark.parametrize("key", [CANONICAL_META_KEY, LEGACY_META_KEY])
def test_metadata_stats_reads_either_key(key):
    assert metadata_stats({key: {"imu_samples": 5}})["imu_samples"] == 5


# ── the two on-device readers ────────────────────────────────────────────────

def _episode(root, name, files, meta=None):
    ep = root / "episodes" / name
    ep.mkdir(parents=True)
    for f in files:
        (ep / f).write_text("{}")
    (ep / "metadata.json").write_text(json.dumps(meta or {}))
    return ep


@pytest.mark.parametrize("imu_file", ["dcam_imu.json", "oakd_imu.json"])
def test_session_reports_has_imu_for_either_layout(tmp_path, imu_file):
    from grabette.session import SessionManager

    _episode(tmp_path, "ep", [imu_file, "raw_video.mp4"])
    info = SessionManager(data_dir=tmp_path)._get_episode_info("ep")
    assert info.has_imu is True


def test_session_reports_no_imu_for_an_imu_less_camera(tmp_path):
    # A Gemini 305 episode has neither file. That is a legitimate absence.
    from grabette.session import SessionManager

    _episode(tmp_path, "ep", ["dcam_left.mp4", "raw_video.mp4"])
    info = SessionManager(data_dir=tmp_path)._get_episode_info("ep")
    assert info.has_imu is False


@pytest.mark.parametrize("meta_key", ["dcam", "oakd"])
def test_session_imu_count_from_either_metadata_key(tmp_path, meta_key):
    from grabette.session import SessionManager

    _episode(tmp_path, "ep", ["dcam_imu.json"], meta={meta_key: {"imu_samples": 2115}})
    info = SessionManager(data_dir=tmp_path)._get_episode_info("ep")
    assert info.imu_sample_count == 2115
