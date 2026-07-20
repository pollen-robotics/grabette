"""Fixtures that build a synthetic OAK+Arducam episode on disk.

The recording checks (grabette_postprocess.checks.recording) are the pipeline's
quality gate: they decide whether a raw episode is fit to feed SLAM and dataset
generation. These fixtures let the tests exercise that gate end to end against a
realistic episode layout, then knock out one file/field at a time to prove each
defect is actually caught.
"""

import json

import av
import numpy as np
import pytest


def _write_video(path, n_frames=4, w=32, h=24, codec="libx264", pix_fmt="yuv420p"):
    """Write a tiny but genuinely decodable video (few KB)."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=30)
        stream.width, stream.height, stream.pix_fmt = w, h, pix_fmt
        for i in range(n_frames):
            arr = np.full((h, w, 3), (i * 20) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)


def _ts_samples(n, *, seq_start=0, period_us=33_000):
    """Frame-timestamp samples carrying the OAK device + host clocks."""
    return [
        {"seq": seq_start + i,
         "device_us": i * period_us,
         "host_ms": i * period_us // 1000}
        for i in range(n)
    ]


def _imu_samples(n, kinds=("accel", "gyro", "rotation")):
    out = []
    for i in range(n):
        for kind in kinds:
            out.append({"kind": kind,
                        "device_us": i * 5000,
                        "host_ms": i * 5,
                        "value": [0.1 * i, 0.2 * i, 9.81]})
    return out


def _angle_samples(n, *, distal_amp=0.5, proximal_amp=0.5):
    """A gripper that actually moves on both joints (value = [distal, proximal])."""
    return {"samples": [
        {"cts": i * 33,
         "value": [distal_amp * np.sin(i / 5), proximal_amp * np.cos(i / 5)]}
        for i in range(n)
    ]}


def build_episode(ep_dir, *, n_frames=4, with_right=True):
    """Write a complete, valid raw episode into ep_dir and return it.

    Every file the recording check requires is present, non-empty and internally
    consistent (left/depth seqs overlap, gripper moves, calib intrinsics
    positive), so check_recording() should report zero errors and zero warnings.
    Tests mutate the result to inject a single defect.
    """
    ep_dir.mkdir(parents=True, exist_ok=True)

    _write_video(ep_dir / "oakd_left.mp4", n_frames)
    _write_video(ep_dir / "raw_video.mp4", n_frames)
    if with_right:
        _write_video(ep_dir / "oakd_right.mp4", n_frames)
    _write_video(ep_dir / "oakd_depth.mkv", n_frames, codec="ffv1", pix_fmt="gray")

    ts = _ts_samples(n_frames)
    (ep_dir / "oakd_left_timestamps.json").write_text(json.dumps({"samples": ts}))
    (ep_dir / "oakd_depth_timestamps.json").write_text(json.dumps({"samples": ts}))

    (ep_dir / "oakd_imu.json").write_text(json.dumps({"samples": _imu_samples(20)}))
    (ep_dir / "angle_data.json").write_text(json.dumps(_angle_samples(30)))
    # Arducam per-frame timestamps (ms), spanning the same wall-clock as the OAK.
    (ep_dir / "frame_timestamps.json").write_text(
        json.dumps([i * 33.0 for i in range(n_frames)]))

    (ep_dir / "oakd_calib_offline.json").write_text(json.dumps({
        "width": 640, "height": 480,
        "fx": 450.0, "fy": 450.0, "cx": 320.0, "cy": 240.0,
        "baseline": 0.075,
        "imu_to_cam": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }))
    return ep_dir


@pytest.fixture
def valid_episode(tmp_path):
    return build_episode(tmp_path / "episode_0000")


@pytest.fixture
def episode_builder():
    """The build_episode function itself, for tests that need custom episodes."""
    return build_episode
