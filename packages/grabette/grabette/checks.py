"""Lightweight local recording check — file existence + metadata only, no heavy deps."""

from __future__ import annotations

import json
from pathlib import Path

# Files the SLAM pipeline requires to run at all.
_SLAM_REQUIRED: list[tuple[str, str]] = [
    ("oakd_left.mp4", "OAK-D left camera video missing (SLAM cannot run)"),
    ("oakd_imu.json", "OAK-D IMU data missing (SLAM needs inertial data)"),
    ("oakd_calib_offline.json", "OAK-D calibration file missing (SLAM accuracy impacted)"),
]

# Files needed for the dataset observation stream (not blocking for SLAM itself).
_DATASET_REQUIRED: list[tuple[str, str]] = [
    ("raw_video.mp4", "Arducam observation video missing"),
    ("frame_timestamps.json", "Frame timestamps missing (sync check will fail)"),
]


def check_recording_local(episode_dir: Path) -> dict:
    """Check file completeness for a grabette episode directory.

    Returns {"errors": [...], "warnings": [...], "verdict": "OK"|"WARN"|"ERROR"}.
    Reads only file existence and metadata.json — no numpy/opencv required.
    """
    errors: list[str] = []
    warnings: list[str] = []
    ep = Path(episode_dir)

    for filename, message in _SLAM_REQUIRED:
        if not (ep / filename).exists():
            errors.append(message)

    for filename, message in _DATASET_REQUIRED:
        if not (ep / filename).exists():
            warnings.append(message)

    meta_path = ep / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            frame_count = meta.get("frame_count", 0)
            imu_samples = (
                meta.get("imu_sample_count")
                or meta.get("oakd", {}).get("imu_samples", 0)
            )
            duration = meta.get("duration_seconds", 0.0)

            if frame_count < 10:
                errors.append(
                    f"Too few frames: {frame_count} (episode likely too short or aborted)"
                )
            if imu_samples == 0:
                warnings.append(
                    "No IMU samples recorded (OAK-D may not have been connected)"
                )
            if duration < 1.0:
                warnings.append(f"Very short episode: {duration:.1f}s")
        except (json.JSONDecodeError, KeyError):
            warnings.append("metadata.json unreadable")
    else:
        warnings.append("metadata.json missing")

    if not (ep / "angle_data.json").exists():
        warnings.append("Gripper angle data missing")

    verdict = "ERROR" if errors else ("WARN" if warnings else "OK")
    return {"errors": errors, "warnings": warnings, "verdict": verdict}
