"""Unit tests for grabette.hardware.frames — URDF → per-episode frame transforms.

frames.json is the contract that lets a downstream consumer re-express SLAM poses
(produced in the oak_l frame) in the primary camera frame. A sign/compose error
here silently corrupts every dataset's geometry, so the rotation math and the
rigid-inverse composition are worth pinning.
"""

import numpy as np

from grabette.hardware.frames import (
    _pose_to_matrix,
    _rpy_to_rotation,
    build_frames_payload,
)

URDF = """<?xml version="1.0"?>
<robot name="test">
  <joint name="oak_l_frame" type="fixed">
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="camera_frame" type="fixed">
    <origin xyz="2 0 0" rpy="0 0 1.5707963267948966"/>
  </joint>
  <joint name="no_origin_frame" type="fixed"/>
</robot>
"""


def test_rpy_to_rotation_identity():
    """Zero roll/pitch/yaw yields the identity rotation."""
    np.testing.assert_allclose(_rpy_to_rotation((0, 0, 0)), np.eye(3), atol=1e-12)


def test_rpy_to_rotation_yaw_90():
    """A 90° yaw builds the expected rotation about Z."""
    R = _rpy_to_rotation((0, 0, np.pi / 2))
    np.testing.assert_allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9)


def test_pose_to_matrix_places_translation():
    """xyz lands in the translation column and the homogeneous row is [0,0,0,1]."""
    T = _pose_to_matrix((1.0, 2.0, 3.0), (0, 0, 0))
    np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_build_frames_payload_composition(tmp_path):
    """T_camera_in_oak_l = inv(T_oak_l) @ T_camera, including the carried rotation."""
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(URDF)
    payload = build_frames_payload(urdf)

    assert payload["parent_link"] == "grip_r"
    assert set(payload["frames_in_grip_r"]) == {"oak_l", "camera"}

    # T_camera_in_oak_l = inv(T_oak_l) @ T_camera.
    # oak_l at x=1 (identity rot), camera at x=2 -> camera sits at x=1 in oak_l.
    T = np.array(payload["T_camera_in_oak_l"])
    np.testing.assert_allclose(T[:3, 3], [1.0, 0.0, 0.0], atol=1e-9)
    # camera carried a 90° yaw; oak_l none -> the relative rotation is that yaw.
    np.testing.assert_allclose(T[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9)


def test_build_frames_payload_none_when_frame_absent(tmp_path):
    """Missing the camera frame leaves T_camera_in_oak_l as None (can't compose)."""
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("""<?xml version="1.0"?>
<robot name="test">
  <joint name="oak_l_frame" type="fixed"><origin xyz="1 0 0" rpy="0 0 0"/></joint>
</robot>""")
    payload = build_frames_payload(urdf)
    # No camera frame -> the convenience transform can't be built.
    assert payload["T_camera_in_oak_l"] is None
    assert "oak_l" in payload["frames_in_grip_r"]
