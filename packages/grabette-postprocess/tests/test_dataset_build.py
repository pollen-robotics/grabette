"""Tier B — functional build_dataset round-trip against the installed lerobot.

Fabricates a tiny synthetic episode (two 4-frame videos + timestamps +
trajectory CSV + angles), runs dataset.build_dataset (the real create ->
add_frame -> save_episode -> finalize flow, incl. rgb_encoder h264 encoding),
then reloads the produced LeRobotDataset and asserts schema / counts / values.

This is the functional guard for a lerobot bump: if 0.x.0 breaks the builder,
this fails where the API-surface test can't. Skipped when lerobot is absent.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("lerobot", reason="lerobot not installed (fast lane)")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from grabette_postprocess.dataset import build_dataset  # noqa: E402

N = 4
TS_MS = [0, 33, 66, 100]
COLS = ["frame_idx", "timestamp", "state", "is_lost", "is_keyframe",
        "x", "y", "z", "q_x", "q_y", "q_z", "q_w"]


def _write_video(path, n, w=128, h=96):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    assert vw.isOpened(), f"could not open VideoWriter for {path}"
    for i in range(n):
        frame = np.full((h, w, 3), i * 30, dtype=np.uint8)  # distinct per frame
        vw.write(frame)
    vw.release()


def _make_episode(ep_dir):
    ep_dir.mkdir(parents=True)
    _write_video(ep_dir / "raw_video.mp4", N)
    _write_video(ep_dir / "oakd_left.mp4", N)
    (ep_dir / "frame_timestamps.json").write_text(json.dumps(TS_MS))
    (ep_dir / "oakd_left_timestamps.json").write_text(
        json.dumps({"samples": [{"host_ms": t} for t in TS_MS]}))
    (ep_dir / "angle_data.json").write_text(json.dumps({"samples": [
        {"cts": t, "value": [0.1 * i, 0.2 * i]} for i, t in enumerate(TS_MS)]}))
    # trajectory: 4 tracked frames, moving along +x, identity rotation
    rows = [[i, TS_MS[i] / 1000.0, "OK", 0, 0, 0.01 * i, 0.0, 0.0, 0, 0, 0, 1]
            for i in range(N)]
    pd.DataFrame(rows, columns=COLS).to_csv(ep_dir / "camera_trajectory.csv", index=False)
    return ep_dir


def test_build_dataset_roundtrip(tmp_path):
    ep = _make_episode(tmp_path / "ep0")
    root = tmp_path / "ds"
    repo_id = "test/grabette_synthetic"

    build_dataset(repo_id, [ep], task="pick", fps=30, root=root)

    # Reload the produced dataset and assert schema + counts + values.
    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=root)

    feats = ds.meta.features
    assert feats["observation.images.cam0"]["dtype"] == "video"
    assert feats["observation.images.cam1"]["dtype"] == "video"
    assert tuple(feats["action"]["shape"]) == (8,)
    assert "is_lost" in feats

    assert ds.meta.total_episodes == 1
    assert ds.meta.total_frames == N
    assert ds.fps == 30

    # action[0] = [x,y,z, ax,ay,az, proximal, distal]; frame 0 -> zeros
    sample = ds[0]
    action = np.asarray(sample["action"]).ravel()
    assert action.shape == (8,)
    np.testing.assert_allclose(action[:6], 0.0, atol=1e-5)  # first pose = origin+identity
