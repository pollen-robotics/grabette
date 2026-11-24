"""Canonical names for the depth-camera files an episode contains.

Recordings used to be named after the hardware (`oakd_left.mp4`, `oak_mask.png`),
which stopped being true when the Orbbec Gemini 305 became a supported camera —
a Gemini recording was emitting files that claimed to be an OAK-D's. The prefix
is now `dcam_` ("depth camera"), honest whichever camera produced the episode.

Writers use the `DCAM_*` constants. The two places that *read* an existing
episode (`session.py` for the dashboard, `replay.py` for playback) go through
`resolve()`, because episodes recorded before the rename still use `oakd_*` and
are not being rewritten.

Deliberately duplicated from `grabette_postprocess.episode_files` in the
grabette-slam repo: the two packages live in separate repositories and neither
depends on the other, so a shared module would mean a new cross-repo dependency
for one table of twelve strings. Keep the two in step.
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


def camera_info(meta: dict) -> dict:
    """Which depth camera recorded this episode, from metadata.json.

    `{}` for episodes recorded before this was captured.
    """
    return meta.get("depth_camera") or {}


def metadata_stats(meta: dict) -> dict:
    """Per-stream capture stats from metadata.json, under either key."""
    return meta.get(CANONICAL_META_KEY) or meta.get(LEGACY_META_KEY) or {}


# Optional LeRobot streams, named as in grabette-slam's `keep` option.
FEATURE_KEYS = ("depth", "right", "imu")
UNKNOWN_CAMERA = "unknown"


def depth_camera_model(ep_dir: Path, fallback: str | None = None) -> str:
    """Depth camera that recorded the episode. Same rules as grabette-slam."""
    import json

    ep_dir = Path(ep_dir)
    meta_path = ep_dir / "metadata.json"
    if meta_path.is_file():
        try:
            model = camera_info(json.loads(meta_path.read_text())).get("model")
        except (OSError, ValueError):
            model = None
        if model:
            return str(model)
    calib_path = resolve(ep_dir, DCAM_CALIB)
    if calib_path.is_file():
        try:
            calib = json.loads(calib_path.read_text())
        except (OSError, ValueError):
            calib = None
        if isinstance(calib, dict):
            if calib.get("productName"):
                return "oakd"
            if "gemini" in str(calib.get("name", "")).lower():
                return "gemini305"
    if fallback:
        return fallback
    if resolve(ep_dir, DCAM_IMU).is_file() or resolve(ep_dir, DCAM_RIGHT).is_file():
        return "oakd"
    return UNKNOWN_CAMERA


def video_aligned(meta: dict) -> bool:
    """False when metadata.json marks the video as not matched to its capture times."""
    alignment = meta.get("video_timestamp_alignment")
    return (meta.get("valid", True) is not False
            and not (isinstance(alignment, dict) and alignment.get("valid") is False))


def invalid_reasons(ep_dir: Path) -> list[str]:
    """Why the episode cannot go into a dataset, from its metadata.json ([] = valid)."""
    import json

    try:
        meta = json.loads((Path(ep_dir) / "metadata.json").read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(meta, dict):
        return []
    if meta.get("stop_error"):
        return ["stop_failed"]
    return [] if video_aligned(meta) else ["video_sync"]


def episode_features(ep_dir: Path, camera_fallback: str | None = None) -> dict:
    """{"camera", "streams"[, "invalid"]}: what the LeRobot conversion can keep."""
    ep_dir = Path(ep_dir)
    present = {
        "depth": ((resolve(ep_dir, DCAM_DEPTH_VIDEO).is_file()
                   or resolve(ep_dir, DCAM_DEPTH_DIR).is_dir())
                  and resolve(ep_dir, DCAM_DEPTH_TS).is_file()),
        "right": (resolve(ep_dir, DCAM_RIGHT).is_file()
                  and resolve(ep_dir, DCAM_RIGHT_TS).is_file()),
        "imu": resolve(ep_dir, DCAM_IMU).is_file(),
    }
    features = {"camera": depth_camera_model(ep_dir, camera_fallback),
                "streams": [k for k in FEATURE_KEYS if present[k]]}
    invalid = invalid_reasons(ep_dir)
    if invalid:
        features["invalid"] = invalid
    return features
