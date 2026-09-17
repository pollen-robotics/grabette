"""IMU JSON output writer for UMI SLAM compatibility.

Ported from grabette-capture/grabette_capture/output.py.
"""

import json
import os
from pathlib import Path
from typing import TypedDict


def write_json_atomic(path: Path, payload) -> None:
    """Write `payload` as JSON so the file is never observed half-written.

    A plain write_text() truncates the target on open, so an interruption
    between that and the flush — power cut on the Pi, SD card full — leaves a
    file that EXISTS but is empty. metadata.json is written last precisely so
    that its presence means "episode fully saved", and a non-atomic write broke
    that promise: a 0-byte metadata.json read back as a JSONDecodeError that
    500'd GET /api/tasks and blanked the whole dashboard. Writing to a sibling
    temp file and renaming makes the swap atomic on POSIX, so the reader sees
    either no file or a complete one.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class IMUSampleDict(TypedDict):
    value: list[float]
    cts: float


def write_imu_json(
    accel_samples: list[IMUSampleDict],
    gyro_samples: list[IMUSampleDict],
    fps: float,
    output_path: Path,
    angle_samples: list[IMUSampleDict] | None = None,
) -> None:
    """Write IMU data in UMI SLAM-compatible JSON format.

    Args:
        accel_samples: Accelerometer samples with 'cts' (ms) and 'value' [ax, ay, az] in m/s².
        gyro_samples: Gyroscope samples with 'cts' (ms) and 'value' [gx, gy, gz] in rad/s.
        fps: Video frame rate (used by SLAM to compute frame timestamps).
        output_path: Output file path for imu_data.json.
        angle_samples: Optional angle sensor samples with 'cts' (ms) and 'value' [proximal, distal] in rad.
    """
    streams = {
        "ACCL": {
            "name": "Accelerometer",
            "units": "m/s2",
            "samples": accel_samples,
        },
        "GYRO": {
            "name": "Gyroscope",
            "units": "rad/s",
            "samples": gyro_samples,
        },
    }

    if angle_samples:
        streams["ANGL"] = {
            "name": "Angle",
            "units": "rad",
            "samples": angle_samples,
        }

    imu_json = {
        "frames/second": fps,
        "1": {
            "streams": streams,
        },
    }

    with open(output_path, "w") as f:
        json.dump(imu_json, f, indent=2)
