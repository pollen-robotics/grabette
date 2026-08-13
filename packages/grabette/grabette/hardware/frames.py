"""URDF frame extraction for per-episode metadata.

Reads a URDF, extracts the fixed-joint origins for the frames a downstream
consumer typically wants (camera, oak_l, oak_r, gripper_center, thumb_tip),
and composes the transform that expresses the primary RPi camera pose in
the OAK-D left frame (the SLAM output frame).

Consumer contract for the per-episode `frames.json`:

    {
      "urdf_source":       "grabette_right/robot.urdf",
      "parent_link":       "grip_r",
      "frames_in_grip_r":  { "camera": [[4x4]], "oak_l": [[4x4]], ... },
      "T_camera_in_oak_l": [[4x4]],
      "note":              "..."
    }

All 4x4 matrices are row-major homogeneous transforms in the URDF unit
convention (meters, radians). `frames_in_grip_r["frame_x"]` is the pose of
`frame_x` expressed in the `grip_r` frame — i.e. p_grip = T @ p_frame_x.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# The set of URDF frames we surface in the per-episode payload. Keys are
# the labels used in the output JSON; values are the URDF joint names.
_FRAMES_OF_INTEREST = {
    "camera":         "camera_frame",
    "oak_l":          "oak_l_frame",
    "oak_r":          "oak_r_frame",
    "gripper_center": "gripper_center_frame",
    "thumb_tip":      "thumb_tip_frame",
}


def _rpy_to_rotation(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw -> 3x3 rotation matrix.

    URDF convention (fixed / extrinsic axes): first roll around X, then pitch
    around Y, then yaw around Z. Equivalent compound rotation is Rz*Ry*Rx.
    """
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float64)


def _pose_to_matrix(xyz: tuple[float, float, float],
                    rpy: tuple[float, float, float]) -> np.ndarray:
    """(xyz, rpy) URDF origin -> 4x4 homogeneous transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_to_rotation(rpy)
    T[:3, 3] = xyz
    return T


def _read_urdf_joint_origins(
    urdf_path: Path,
) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Parse a URDF and return {joint_name: (xyz, rpy)} for every joint that
    has an <origin> child. Missing attributes default to zeros per URDF spec."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    out: dict[str, tuple] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        origin = joint.find("origin")
        if name is None or origin is None:
            continue
        xyz = tuple(float(x) for x in origin.get("xyz", "0 0 0").split())
        rpy = tuple(float(x) for x in origin.get("rpy", "0 0 0").split())
        out[name] = (xyz, rpy)
    return out


def build_frames_payload(urdf_path: Path) -> dict:
    """Read a URDF and produce the JSON-serializable per-episode frames payload.

    The frames we surface all share `grip_r` as their URDF parent link, so
    `frames_in_grip_r[X]` directly equals `T_X_in_grip_r`. `T_camera_in_oak_l`
    is precomposed as a convenience for consumers who want the primary
    camera pose in the SLAM output frame:

        T_camera_in_oak_l = inv(T_oak_l_in_grip_r) @ T_camera_in_grip_r
    """
    origins = _read_urdf_joint_origins(urdf_path)

    frames_in_grip: dict[str, list[list[float]]] = {}
    matrices: dict[str, np.ndarray] = {}
    for label, joint_name in _FRAMES_OF_INTEREST.items():
        if joint_name not in origins:
            continue
        M = _pose_to_matrix(*origins[joint_name])
        matrices[label] = M
        frames_in_grip[label] = M.tolist()

    T_camera_in_oak_l: list[list[float]] | None = None
    if "camera" in matrices and "oak_l" in matrices:
        # Rigid-transform inverse: R^T and -R^T @ t.
        T_oak_l = matrices["oak_l"]
        R_T = T_oak_l[:3, :3].T
        t = T_oak_l[:3, 3]
        T_oak_l_inv = np.eye(4)
        T_oak_l_inv[:3, :3] = R_T
        T_oak_l_inv[:3, 3] = -R_T @ t
        T_camera_in_oak_l = (T_oak_l_inv @ matrices["camera"]).tolist()

    return {
        "urdf_source": str(urdf_path.name if urdf_path.parent.name == ""
                           else Path(urdf_path.parent.name) / urdf_path.name),
        "parent_link": "grip_r",
        "frames_in_grip_r": frames_in_grip,
        "T_camera_in_oak_l": T_camera_in_oak_l,
        "note": (
            "4x4 row-major homogeneous transforms, meters + radians. "
            "frames_in_grip_r[X] = T_X_in_grip_r (pose of frame X in the "
            "grip_r link frame). T_camera_in_oak_l = "
            "inv(T_oak_l_in_grip_r) @ T_camera_in_grip_r — provided as a "
            "convenience so SLAM poses (produced in the oak_l frame) can be "
            "re-expressed in the primary RPi camera frame without URDF parsing."
        ),
    }
