"""IMU JSON output writer for UMI SLAM compatibility.

Ported from grabette-capture/grabette_capture/output.py.
"""

import json
from pathlib import Path
from typing import TypedDict


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
