"""Canonical names for the depth-camera files an episode contains.

Recordings used to be named after the hardware (`oakd_left.mp4`, `oak_mask.png`),
which stopped being true when the Orbbec Gemini 305 became a supported camera —
a Gemini recording was emitting files that claimed to be an OAK-D's. The prefix
is now `dcam_` ("depth camera"), honest whichever camera produced the episode.

Writers use the `DCAM_*` constants. The two places that *read* an existing
episode (`session.py` for the dashboard, `replay.py` for playback) go through
`resolve()`, because episodes recorded before the rename still use `oakd_*` and
are not being rewritten.

Deliberately duplicated from `grabette_postprocess.episode_files`: the two
packages are independent distributions (neither depends on the other, and the
SLAM Space vendors only postprocess), so a shared module would mean a new
dependency edge for one table of twelve strings. Keep the two in step.
"""

from __future__ import annotations

from pathlib import Path

# Written by the capture classes.
DCAM_LEFT = "dcam_left.mp4"
DCAM_RIGHT = "dcam_right.mp4"
DCAM_DEPTH_VIDEO = "dcam_depth.mkv"
DCAM_DEPTH_DIR = "dcam_depth"
DCAM_IMU = "dcam_imu.json"
DCAM_CALIB = "dcam_calib.json"
DCAM_CALIB_OFFLINE = "dcam_calib_offline.json"
DCAM_LEFT_TS = "dcam_left_timestamps.json"
DCAM_RIGHT_TS = "dcam_right_timestamps.json"
DCAM_DEPTH_TS = "dcam_depth_timestamps.json"
DCAM_CLOCK_PAIRS = "dcam_clock_pairs.json"
DCAM_MASK = "dcam_mask.png"

# Per-stream capture stats inside metadata.json. Older episodes use "oakd".
CANONICAL_META_KEY = "dcam"
LEGACY_META_KEY = "oakd"

# Canonical -> the legacy name it replaced. The mask is the odd one out: it was
# `oak_mask.png`, not `oakd_mask.png`.
LEGACY_NAMES = {
    DCAM_LEFT: "oakd_left.mp4",
    DCAM_RIGHT: "oakd_right.mp4",
    DCAM_DEPTH_VIDEO: "oakd_depth.mkv",
    DCAM_DEPTH_DIR: "oakd_depth",
    DCAM_IMU: "oakd_imu.json",
    DCAM_CALIB: "oakd_calib.json",
    DCAM_CALIB_OFFLINE: "oakd_calib_offline.json",
    DCAM_LEFT_TS: "oakd_left_timestamps.json",
    DCAM_RIGHT_TS: "oakd_right_timestamps.json",
    DCAM_DEPTH_TS: "oakd_depth_timestamps.json",
    DCAM_CLOCK_PAIRS: "oakd_clock_pairs.json",
    DCAM_MASK: "oak_mask.png",
}


def legacy_name(name: str) -> str:
    """The pre-`dcam_` name for a canonical one (itself, if there isn't one)."""
    return LEGACY_NAMES.get(name, name)


def resolve(ep_dir: Path, name: str) -> Path:
    """Path to `name` in `ep_dir`, accepting a legacy-named episode.

    Prefers the canonical name, falls back to the legacy one, and returns the
    canonical path when neither exists so "missing X" messages name the file
    people should be producing now.
    """
    ep_dir = Path(ep_dir)
    canonical = ep_dir / name
    if canonical.exists():
        return canonical
    legacy = ep_dir / legacy_name(name)
    if legacy.exists():
        return legacy
    return canonical


def metadata_stats(meta: dict) -> dict:
    """Per-stream capture stats from metadata.json, under either key."""
    return meta.get(CANONICAL_META_KEY) or meta.get(LEGACY_META_KEY) or {}
