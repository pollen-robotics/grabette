"""6D continuous rotation representation (Zhou et al., CVPR 2019).

Encodes orientation as the first two columns of the rotation matrix (6 values).
No singularities, smooth gradients for neural networks.
"""

import numpy as np


def rotation_matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to 6D representation.

    Args:
        matrix: shape (3, 3) or (N, 3, 3).

    Returns:
        6D vector(s), shape (6,) or (N, 6).
    """
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rot_6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation representation to a 3x3 rotation matrix.

    Uses Gram-Schmidt orthogonalization to recover the third column.

    Args:
        rot_6d: shape (6,) or (N, 6).

    Returns:
        Rotation matrix, shape (3, 3) or (N, 3, 3).
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]

    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack([b1, b2, b3], axis=-2)
