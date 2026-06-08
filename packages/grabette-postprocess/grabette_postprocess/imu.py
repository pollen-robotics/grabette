"""IMU resampling for ORB-SLAM3.

ORB-SLAM3 requires uniformly spaced IMU measurements. Its noise model assumes
dt = 1/IMU.Frequency for every step. Non-uniform timestamps cause covariance
errors and tracking failure.

Ported from universal_manipulation_interface/scripts/resample_imu.py.
"""

import json
from pathlib import Path

import numpy as np


def deduplicate_samples(samples: list[dict]) -> list[dict]:
    """Remove consecutive samples with identical values (stale sensor reads)."""
    if not samples:
        return samples
    deduped = [samples[0]]
    for s in samples[1:]:
        if s['value'] != deduped[-1]['value']:
            deduped.append(s)
    return deduped


def resample_stream(samples: list[dict], target_rate_hz: int = 200) -> list[dict]:
    """Resample IMU stream to uniform spacing using linear interpolation.

    Args:
        samples: list of {"cts": float_ms, "value": [x, y, z]}
        target_rate_hz: target rate (e.g. 200 for 5ms spacing)

    Returns:
        list of resampled samples with uniform cts spacing
    """
    if len(samples) < 2:
        return samples

    dt_ms = 1000.0 / target_rate_hz

    cts = np.array([s['cts'] for s in samples])
    values = np.array([s['value'] for s in samples])
    n_axes = values.shape[1]

    # Build uniform grid starting from first timestamp
    uniform_cts = np.arange(cts[0], cts[-1], dt_ms)

    # Interpolate each axis
    resampled_values = np.zeros((len(uniform_cts), n_axes))
    for axis in range(n_axes):
        resampled_values[:, axis] = np.interp(uniform_cts, cts, values[:, axis])

    return [
        {'cts': float(uniform_cts[i]), 'value': resampled_values[i].tolist()}
        for i in range(len(uniform_cts))
    ]


def prepare_imu_for_slam(imu_json_path: Path, output_path: Path) -> Path:
    """Load raw IMU JSON, resample ACCL+GYRO to uniform 200Hz, strip ANGL, save.

    Args:
        imu_json_path: path to raw imu_data.json
        output_path: path for resampled output

    Returns:
        output_path
    """
    with open(imu_json_path) as f:
        data = json.load(f)

    streams = data['1']['streams']

    for stream_name in ['ACCL', 'GYRO']:
        if stream_name not in streams:
            continue
        samples = streams[stream_name]['samples']
        samples = deduplicate_samples(samples)
        resampled = resample_stream(samples, 200)
        streams[stream_name]['samples'] = resampled
        print(f"  {stream_name}: {len(samples)} deduped -> {len(resampled)} resampled")

    # ANGL not needed for SLAM
    if 'ANGL' in streams:
        del streams['ANGL']

    with open(output_path, 'w') as f:
        json.dump({"1": {"streams": streams}}, f)

    return output_path
