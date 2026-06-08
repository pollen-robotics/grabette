"""Grabette device mask generation.

Generates a grayscale mask image that blacks out the device body (cylindrical
housing and arm) visible in the bottom-right of the camera frame.

Polygon coordinates defined at reference resolution 1296x972, auto-scaled
to any target resolution.

Ported from universal_manipulation_interface/umi/common/cv_util.py.
"""

import cv2
import numpy as np

# Reference resolution the polygon was defined at
_REF_W = 1296
_REF_H = 972

# Device body polygon at reference resolution (x, y)
_DEVICE_BODY_PTS = np.array([
    [120, _REF_H],   # bottom, start of device edge
    [280, 750],       # left side of curved arm
    [1030, 610],      # top of device body
    [1160, 780],      # top-right corner area
    [_REF_W, 780],    # right edge
    [_REF_W, _REF_H], # bottom-right corner
], dtype=np.int32)


def generate_mask(width: int, height: int) -> np.ndarray:
    """Generate grayscale mask for grabette device body.

    Args:
        width: target image width
        height: target image height

    Returns:
        uint8 grayscale mask: 255 = masked region, 0 = keep
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Scale polygon from reference to target resolution
    scale_x = width / _REF_W
    scale_y = height / _REF_H
    pts = _DEVICE_BODY_PTS.astype(np.float64).copy()
    pts[:, 0] *= scale_x
    pts[:, 1] *= scale_y
    pts = np.round(pts).astype(np.int32)

    cv2.fillPoly(mask, [pts], color=255)
    return mask
