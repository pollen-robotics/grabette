"""Names of the depth-camera files inside a raw episode, old and new.

Recordings used to be named after the hardware — `oakd_left.mp4`, `oakd_imu.json`,
`oak_mask.png` — which stopped being true once the Orbbec Gemini 305 became a
supported camera: a Gemini recording was writing files claiming to be an OAK-D's.
The canonical prefix is now `dcam_` ("depth camera"), which is honest whichever
camera produced the episode.

**Readers must go through `resolve()`**, never a bare literal. Every episode
recorded before this change — including the ones already on the Hub — still uses
`oakd_*`, and those datasets are not being rewritten. `resolve()` prefers the
canonical name and falls back to the legacy one, so both layouts keep working.

The fallback is deliberate debt with an exit: once no `oakd_*` episodes are still
in circulation, drop `LEGACY_NAMES` and the fallback branch, and `resolve()`
becomes a plain path join.
"""

from __future__ import annotations

from pathlib import Path

CANONICAL_PREFIX = "dcam_"

# Canonical name -> the legacy name it replaced. Most are a prefix swap; the mask
# is the exception, because it was `oak_mask.png` rather than `oakd_mask.png`.
LEGACY_NAMES = {
    "dcam_left.mp4": "oakd_left.mp4",
    "dcam_right.mp4": "oakd_right.mp4",
    "dcam_depth.mkv": "oakd_depth.mkv",
    "dcam_depth": "oakd_depth",                      # per-frame PNG directory
    "dcam_imu.json": "oakd_imu.json",
    "dcam_calib.json": "oakd_calib.json",
    "dcam_calib_offline.json": "oakd_calib_offline.json",
    "dcam_left_timestamps.json": "oakd_left_timestamps.json",
    "dcam_right_timestamps.json": "oakd_right_timestamps.json",
    "dcam_depth_timestamps.json": "oakd_depth_timestamps.json",
    "dcam_clock_pairs.json": "oakd_clock_pairs.json",
    "dcam_mask.png": "oak_mask.png",
}

# metadata.json grew up with the same problem: the per-stream stats live under an
# "oakd" key. New recordings write "dcam"; `metadata_stats()` reads either.
CANONICAL_META_KEY = "dcam"
LEGACY_META_KEY = "oakd"


def legacy_name(name: str) -> str:
    """The pre-`dcam_` name for a canonical one (itself, if there isn't one)."""
    return LEGACY_NAMES.get(name, name)


def resolve(ep_dir: Path, name: str) -> Path:
    """Path to `name` inside `ep_dir`, accepting a legacy-named episode.

    Returns the canonical path when it exists, otherwise the legacy path when
    *that* exists. When neither exists the canonical path is returned, so callers
    reporting "missing <file>" name the file people should be producing now.
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


def camera_info(meta: dict) -> dict:
    """Which depth camera recorded this episode, from metadata.json.

    Returns `{}` for episodes recorded before this was captured — honest, since
    those genuinely do not say. Callers that need a label can fall back to
    sniffing `dcam_calib.json`, whose identity fields are vendor-specific
    (`productName` on an OAK-D, `name` on an Orbbec).
    """
    return meta.get("depth_camera") or {}


def describe_camera(meta: dict) -> str:
    """One-line camera description for reports; "unknown" when unrecorded."""
    info = camera_info(meta)
    if not info:
        return "unknown"
    label = info.get("name") or info.get("model") or "unknown"
    serial = info.get("serial")
    return f"{label} ({serial})" if serial else label


def is_legacy_episode(ep_dir: Path) -> bool:
    """True when this episode uses the old `oakd_*` layout.

    Handy for reporting which convention an episode is in; not needed to read it.
    """
    ep_dir = Path(ep_dir)
    return not (ep_dir / "dcam_left.mp4").exists() and (ep_dir / "oakd_left.mp4").exists()
