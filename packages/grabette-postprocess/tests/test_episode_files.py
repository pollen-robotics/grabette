"""Both episode layouts must stay readable.

Recordings are named `dcam_*` now. Every episode made before that — including the
ones already on the Hub, which are not being rewritten — uses `oakd_*`. If the
fallback in `episode_files.resolve()` ever breaks, those datasets stop processing
silently: `convert_episode` would report a missing input for a file that is
sitting right there under its old name.

The existing tests in test_check_recording.py and test_dataset_build.py are
written against the LEGACY names on purpose, so they cover that direction too.
This file pins the resolver itself and the canonical direction.
"""
import json

import pytest

from grabette_postprocess.episode_files import (
    CANONICAL_META_KEY,
    LEGACY_META_KEY,
    LEGACY_NAMES,
    is_legacy_episode,
    legacy_name,
    metadata_stats,
    resolve,
)


# ── the mapping ──────────────────────────────────────────────────────────────

def test_every_canonical_name_is_dcam_prefixed():
    for canonical in LEGACY_NAMES:
        assert canonical.startswith("dcam_"), canonical


def test_the_mask_is_the_one_irregular_rename():
    # oak_mask.png, not oakd_mask.png — the reason a blind prefix swap is wrong.
    assert legacy_name("dcam_mask.png") == "oak_mask.png"


def test_legacy_name_passes_through_unknown_names():
    assert legacy_name("raw_video.mp4") == "raw_video.mp4"


# ── resolve() ────────────────────────────────────────────────────────────────

def test_resolve_prefers_the_canonical_file(tmp_path):
    (tmp_path / "dcam_left.mp4").write_text("new")
    (tmp_path / "oakd_left.mp4").write_text("old")
    assert resolve(tmp_path, "dcam_left.mp4").name == "dcam_left.mp4"


def test_resolve_falls_back_to_legacy(tmp_path):
    (tmp_path / "oakd_left.mp4").write_text("old")
    assert resolve(tmp_path, "dcam_left.mp4").name == "oakd_left.mp4"


def test_resolve_falls_back_for_the_mask(tmp_path):
    (tmp_path / "oak_mask.png").write_text("old")
    assert resolve(tmp_path, "dcam_mask.png").name == "oak_mask.png"


def test_resolve_names_the_canonical_file_when_neither_exists(tmp_path):
    # So "missing X" messages tell people what to produce now.
    assert resolve(tmp_path, "dcam_imu.json").name == "dcam_imu.json"


def test_resolve_handles_the_depth_directory(tmp_path):
    (tmp_path / "oakd_depth").mkdir()
    assert resolve(tmp_path, "dcam_depth").name == "oakd_depth"


def test_is_legacy_episode(tmp_path):
    assert is_legacy_episode(tmp_path) is False       # empty: not legacy, just empty
    (tmp_path / "oakd_left.mp4").write_text("x")
    assert is_legacy_episode(tmp_path) is True
    (tmp_path / "dcam_left.mp4").write_text("x")
    assert is_legacy_episode(tmp_path) is False


# ── metadata.json stats key ──────────────────────────────────────────────────

@pytest.mark.parametrize("key", [CANONICAL_META_KEY, LEGACY_META_KEY])
def test_metadata_stats_reads_either_key(key):
    assert metadata_stats({key: {"imu_samples": 7}})["imu_samples"] == 7


def test_metadata_stats_prefers_canonical():
    meta = {CANONICAL_META_KEY: {"imu_samples": 1}, LEGACY_META_KEY: {"imu_samples": 2}}
    assert metadata_stats(meta)["imu_samples"] == 1


def test_metadata_stats_empty_when_absent():
    assert metadata_stats({}) == {}


# ── discovery must see both layouts ──────────────────────────────────────────

@pytest.mark.parametrize("anchor_file", ["dcam_left.mp4", "oakd_left.mp4"])
def test_find_episodes_sees_both_layouts(tmp_path, anchor_file):
    from grabette_postprocess.episode_manager import find_episodes

    ep = tmp_path / "20260101_000000"
    ep.mkdir()
    (ep / anchor_file).write_text("x")
    assert find_episodes(tmp_path) == [ep]


def test_find_episodes_on_a_single_episode_dir(tmp_path):
    from grabette_postprocess.episode_manager import find_episodes

    (tmp_path / "oakd_left.mp4").write_text("x")
    assert find_episodes(tmp_path) == [tmp_path]


# ── the checks accept a canonically-named episode ────────────────────────────

def test_check_recording_accepts_canonical_names(tmp_path):
    from grabette_postprocess.checks.recording import _check_calib, _check_imu

    status = {"errors": [], "warnings": [], "info": []}
    (tmp_path / "dcam_imu.json").write_text(json.dumps({"samples": [
        {"kind": "accel", "value": [0, 0, 9.8]},
        {"kind": "gyro", "value": [0, 0, 0]},
    ]}))
    _check_imu(tmp_path, status)
    assert not status["errors"], status["errors"]

    (tmp_path / "dcam_calib_offline.json").write_text(json.dumps({
        "width": 640, "height": 400, "fx": 311.4, "fy": 311.4,
        "cx": 318.75, "cy": 196.25, "baseline": 0.018,
    }))
    _check_calib(tmp_path, status)
    assert not status["errors"], status["errors"]
