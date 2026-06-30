"""6D continuous rotation helpers (Zhou et al., CVPR 2019), vendored.

These three functions are used by ``convert_dataset.py`` to build the
camera-local delta actions. They are **vendored on purpose**: the equivalents
in the Pollen lerobot fork (``lerobot.utils.rotation.*_numpy`` /
``rotvec_to_rotation_6d``) are fork additions and are NOT present in stock
upstream lerobot, which this package depends on. Vendoring keeps the data
conversion reproducible against any lerobot version.

Convention: the 6D representation is the **first two rows** of the 3x3 rotation
matrix (``matrix[..., :2, :]``), matching the lerobot fork and the simulator's
``rotation_6d_to_matrix``. Verified numerically identical to the fork's
``rotvec_to_rotation_6d`` (max abs diff ~6e-7, float32 precision).
"""

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation


def rotation_matrix_to_rotation_6d_numpy(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) -> 6D representation (..., 6).

    The 6D vector is the first two rows of the matrix, flattened.
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_rotation_matrix_numpy(rot_6d: np.ndarray) -> np.ndarray:
    """6D representation (..., 6) -> rotation matrices (..., 3, 3).

    Recovers the third row via Gram-Schmidt orthogonalization.
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-2)


def rotvec_to_rotation_6d(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vectors (..., 3) -> 6D representation (..., 6).

    Direction is the rotation axis, magnitude is the angle in radians.
    Numerically matches the lerobot fork's torch implementation.
    """
    rv = np.asarray(rotvec, dtype=np.float64)
    matrix = _Rotation.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(*rv.shape[:-1], 3, 3)
    return rotation_matrix_to_rotation_6d_numpy(matrix)
