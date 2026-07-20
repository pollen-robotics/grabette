"""Example unit tests for grabette_postprocess.trajectory.

These exercise the pure data-layer functions (no I/O, no hardware) and serve as
the template for future tests: fast, deterministic, and asserting on numbers we
can reason about by hand. Add new tests alongside these mirroring the source
layout (tests/<package>/test_<module>.py).
"""

import json

import numpy as np
import pandas as pd

from grabette_postprocess.trajectory import (
    interpolate_angles,
    quaternion_to_axis_angle,
    trajectory_to_poses,
)


def test_quaternion_identity_is_zero_rotvec():
    """The identity quaternion maps to the zero rotation vector."""
    # The identity quaternion (x,y,z,w)=(0,0,0,1) is no rotation.
    rotvec = quaternion_to_axis_angle(
        np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]),
    )
    assert rotvec.shape == (1, 3)
    np.testing.assert_allclose(rotvec[0], [0.0, 0.0, 0.0], atol=1e-9)


def test_quaternion_90deg_about_z():
    """A 90° rotation about +Z becomes a rotvec of magnitude pi/2 along +Z."""
    # 90° about +Z -> rotvec of magnitude pi/2 along +Z.
    s = np.sqrt(0.5)
    rotvec = quaternion_to_axis_angle(
        np.array([0.0]), np.array([0.0]), np.array([s]), np.array([s]),
    )
    np.testing.assert_allclose(rotvec[0], [0.0, 0.0, np.pi / 2], atol=1e-6)


def test_trajectory_to_poses_shape_and_position():
    """A trajectory DataFrame becomes an (N,6) float32 pose array with positions preserved."""
    df = pd.DataFrame({
        "x": [1.0, 2.0], "y": [3.0, 4.0], "z": [5.0, 6.0],
        "q_x": [0.0, 0.0], "q_y": [0.0, 0.0],
        "q_z": [0.0, 0.0], "q_w": [1.0, 1.0],
    })
    poses = trajectory_to_poses(df)
    assert poses.shape == (2, 6)
    assert poses.dtype == np.float32
    np.testing.assert_allclose(poses[:, :3], [[1, 3, 5], [2, 4, 6]])
    # Identity quaternions -> zero rotation vectors.
    np.testing.assert_allclose(poses[:, 3:], np.zeros((2, 3)), atol=1e-6)


def test_trajectory_to_poses_empty():
    """An empty trajectory yields a well-formed (0,6) array, not an error."""
    empty = pd.DataFrame(
        columns=["x", "y", "z", "q_x", "q_y", "q_z", "q_w"],
    )
    assert trajectory_to_poses(empty).shape == (0, 6)


def test_interpolate_angles_swaps_and_interpolates(tmp_path):
    """Angles interpolate to query times and return [proximal, distal] (swapped from file order)."""
    # value = [distal, proximal]; interpolate_angles returns [proximal, distal].
    # cts is in milliseconds. Two samples at t=0ms and t=1000ms.
    data = {"samples": [
        {"cts": 0, "value": [0.0, 10.0]},      # distal=0,  proximal=10
        {"cts": 1000, "value": [2.0, 20.0]},   # distal=2,  proximal=20
    ]}
    path = tmp_path / "angle_data.json"
    path.write_text(json.dumps(data))

    # Query at t=0.5s -> halfway between the two samples.
    out = interpolate_angles(path, np.array([0.5]))
    assert out.shape == (1, 2)
    # column 0 = proximal (10->20 => 15), column 1 = distal (0->2 => 1)
    np.testing.assert_allclose(out[0], [15.0, 1.0], atol=1e-6)
