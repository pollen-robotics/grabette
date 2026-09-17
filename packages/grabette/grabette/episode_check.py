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

import json
from pathlib import Path

from grabette.hardware.episode_files import (DCAM_CALIB_OFFLINE, DCAM_DEPTH_DIR,
                                             DCAM_DEPTH_TS, DCAM_DEPTH_VIDEO,
                                             DCAM_IMU, DCAM_LEFT, DCAM_LEFT_TS,
                                             resolve)

# Files every episode must carry for the raw → LeRobot conversion to be possible.
# Mirrors the required inputs of grabette_postprocess.checks.recording:
#   SLAM     — dcam_left.mp4 + timestamps, depth + timestamps, imu, offline calib
#   Dataset  — angle_data.json (gripper), raw_video.mp4 (Arducam)
# Named canonically; episodes recorded before the dcam_ rename carry the oakd_*
# names and are found through resolve(), so both layouts pass this screen.
REQUIRED_FILES = (
    DCAM_LEFT,
    DCAM_LEFT_TS,
    DCAM_DEPTH_TS,
    DCAM_IMU,
    DCAM_CALIB_OFFLINE,
    "angle_data.json",
    "raw_video.mp4",
)

# Depth ships either as a muxed file or as the PNG sequence it is built from
# (stop_recording muxes the directory into the .mkv, so which one is present
# depends on when the episode was recorded). Either satisfies the requirement.
_DEPTH_ALTERNATIVES = (DCAM_DEPTH_VIDEO, DCAM_DEPTH_DIR)


def _present(path: Path) -> bool:
    """A file that exists but is empty is missing for our purposes — a zero-byte
    video or JSON fails upstream exactly like an absent one, and an interrupted
    recording is the usual way to produce it."""
    if path.is_dir():
        return any(path.iterdir())
    return path.is_file() and path.stat().st_size > 0


def _expects_imu(episode_dir: Path) -> bool:
    """Whether the camera that recorded this episode has an IMU at all.

    metadata.json carries `depth_camera.imu`: a model string on the OAK-D, and
    an explicit null on the Gemini 305, which has none. A null therefore means
    "known absent", and the missing IMU file is a property of the hardware
    rather than a broken recording — flagging it would make every 305 episode
    look incomplete.

    Anything unreadable (no metadata, an older episode, a camera that did not
    record the field) keeps the file required, which is the behaviour this
    screen had before a second camera existed.
    """
    meta_path = episode_dir / "metadata.json"
    if not meta_path.is_file():
        return True
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return True
    cam = meta.get("depth_camera")
    if not isinstance(cam, dict) or "imu" not in cam:
        return True
    return cam["imu"] is not None


def missing_files(episode_dir: Path) -> list[str]:
    """The required artifacts this episode lacks, in a stable order ([] = fine).

    Reported under the canonical dcam_ name so the device and the Space speak
    one vocabulary; a legacy oakd_* episode still satisfies the requirement,
    because resolve() accepts either spelling."""
    episode_dir = Path(episode_dir)
    required = [name for name in REQUIRED_FILES
                if name != DCAM_IMU or _expects_imu(episode_dir)]
    missing = [name for name in required
               if not _present(resolve(episode_dir, name))]
    if not any(_present(resolve(episode_dir, alt)) for alt in _DEPTH_ALTERNATIVES):
        missing.append(_DEPTH_ALTERNATIVES[0])
    return sorted(missing)
