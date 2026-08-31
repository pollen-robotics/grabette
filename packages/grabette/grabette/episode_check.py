"""Is a recorded episode complete enough to be convertible?

The SLAM Space runs the authoritative check (grabette_postprocess
checks.recording): it decodes videos, counts samples and looks for stuck joints.
This is the cheap local mirror of the same contract — presence and non-emptiness
only, no decoding — so the device can answer the one question that matters
BEFORE it spends minutes pushing gigabytes to the Hub:

    will this episode be thrown out on the other side?

That question used to be answered far too late. A grabette whose OAK-D never
produced oakd_calib_offline.json uploaded a whole session normally; the Space
then rejected every episode ("missing oakd_calib_offline.json"), reported no
usable recording, and the operator learned about it after the upload, from a
dataset link that 404'd. Checking here turns that into a named, per-episode
reason attached to the upload result, while the take could still be redone.

Deliberately NOT a re-implementation of the full check: anything this passes may
still be rejected upstream (a truncated video, an angle channel that never
moved). It is a fast pre-filter for the failures that are both common and
locally visible, not a second source of truth.
"""

from __future__ import annotations

from pathlib import Path

# Files every episode must carry for the raw → LeRobot conversion to be possible.
# Mirrors the required inputs of grabette_postprocess.checks.recording:
#   SLAM     — oakd_left.mp4 + timestamps, depth + timestamps, imu, offline calib
#   Dataset  — angle_data.json (gripper), raw_video.mp4 (Arducam)
REQUIRED_FILES = (
    "oakd_left.mp4",
    "oakd_left_timestamps.json",
    "oakd_depth_timestamps.json",
    "oakd_imu.json",
    "oakd_calib_offline.json",
    "angle_data.json",
    "raw_video.mp4",
)

# Depth ships either as a muxed file or as the PNG sequence it is built from
# (stop_recording muxes the directory into the .mkv, so which one is present
# depends on when the episode was recorded). Either satisfies the requirement.
_DEPTH_ALTERNATIVES = ("oakd_depth.mkv", "oakd_depth")


def _present(path: Path) -> bool:
    """A file that exists but is empty is missing for our purposes — a zero-byte
    video or JSON fails upstream exactly like an absent one, and an interrupted
    recording is the usual way to produce it."""
    if path.is_dir():
        return any(path.iterdir())
    return path.is_file() and path.stat().st_size > 0


def missing_files(episode_dir: Path) -> list[str]:
    """The required artifacts this episode lacks, in a stable order ([] = fine).

    Names are returned as-is so they can be shown to an operator verbatim: the
    string the Space reports is the string the device reports."""
    episode_dir = Path(episode_dir)
    missing = [name for name in REQUIRED_FILES
               if not _present(episode_dir / name)]
    if not any(_present(episode_dir / alt) for alt in _DEPTH_ALTERNATIVES):
        missing.append(_DEPTH_ALTERNATIVES[0])
    return sorted(missing)
