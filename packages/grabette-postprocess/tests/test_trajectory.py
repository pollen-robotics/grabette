"""Unit tests for the trajectory data/math layer (grabette_postprocess.trajectory).

Pure math + a temp angle_data.json — no SLAM, no video, no lerobot.
"""
import json
import math

import numpy as np
import pandas as pd

from grabette_postprocess.trajectory import (
    interpolate_angles,
    quaternion_to_axis_angle,
    trajectory_to_poses,
)


def test_quaternion_identity_is_zero_rotvec():
    rv = quaternion_to_axis_angle(
        np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0])
    )
    assert rv.shape == (1, 3)
    np.testing.assert_allclose(rv[0], [0, 0, 0], atol=1e-7)


def test_quaternion_90deg_about_z():
    s = math.sin(math.pi / 4)  # == cos(pi/4)
    rv = quaternion_to_axis_angle(
        np.array([0.0]), np.array([0.0]), np.array([s]), np.array([s])
    )
    np.testing.assert_allclose(rv[0], [0, 0, math.pi / 2], atol=1e-6)


def test_quaternion_180deg_about_x():
    rv = quaternion_to_axis_angle(
        np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])
    )
    np.testing.assert_allclose(np.abs(rv[0]), [math.pi, 0, 0], atol=1e-6)


def test_trajectory_to_poses_empty():
    poses = trajectory_to_poses(
        pd.DataFrame(columns=["x", "y", "z", "q_x", "q_y", "q_z", "q_w"])
    )
    assert poses.shape == (0, 6)


def test_trajectory_to_poses_positions_and_identity_rotation():
    df = pd.DataFrame(
        [[1.0, 2.0, 3.0, 0, 0, 0, 1],
         [4.0, 5.0, 6.0, 0, 0, 0, 1]],
        columns=["x", "y", "z", "q_x", "q_y", "q_z", "q_w"],
    )
    poses = trajectory_to_poses(df)
    assert poses.shape == (2, 6)
    assert poses.dtype == np.float32
    np.testing.assert_allclose(poses[:, :3], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(poses[:, 3:], 0, atol=1e-6)


def test_interpolate_angles_swaps_distal_proximal(tmp_path):
    # Stream stores value = [distal, proximal]; interpolate_angles returns
    # [proximal, distal]. distal 0->1, proximal 0->2 over 0..1s.
    (tmp_path / "angle_data.json").write_text(json.dumps({"samples": [
        {"cts": 0, "value": [0.0, 0.0]},
        {"cts": 1000, "value": [1.0, 2.0]},
    ]}))
    out = interpolate_angles(tmp_path / "angle_data.json", np.array([0.0, 0.5, 1.0]))
    assert out.shape == (3, 2)
    # at t=0.5s: proximal = 1.0 (half of 0->2), distal = 0.5 (half of 0->1)
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[1], [1.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(out[2], [2.0, 1.0], atol=1e-6)


def test_interpolate_angles_clamps_out_of_range(tmp_path):
    (tmp_path / "angle_data.json").write_text(json.dumps({"samples": [
        {"cts": 0, "value": [0.0, 0.0]},
        {"cts": 1000, "value": [1.0, 2.0]},
    ]}))
    out = interpolate_angles(tmp_path / "angle_data.json", np.array([-5.0, 99.0]))
    np.testing.assert_allclose(out[0], [0.0, 0.0], atol=1e-6)   # clamp to first
    np.testing.assert_allclose(out[1], [2.0, 1.0], atol=1e-6)   # clamp to last
