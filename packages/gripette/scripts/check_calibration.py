"""Camera calibration check — captures a frame and overlays the undistorted grid.

Verifies that camera intrinsics from the calibration file match the actual camera.
Saves two images:
  - calibration_raw.jpg      — original fisheye frame
  - calibration_undistort.jpg — undistorted frame (straight lines should be straight)

Run on the Pi:
    python scripts/check_calibration.py [calibration.json]
"""

import json
import sys
import time

import numpy as np


def load_calibration(path):
    """Load camera intrinsics from OpenICC JSON format."""
    with open(path) as f:
        data = json.load(f)

    intr = data["intrinsics"]
    fx = intr["focal_length"]
    fy = fx * intr["aspect_ratio"]
    cx = intr["principal_pt_x"]
    cy = intr["principal_pt_y"]

    K = np.array([
        [fx, intr.get("skew", 0), cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    # KannalaBrandt8 / fisheye: 4 radial distortion coefficients
    D = np.array([
        intr["radial_distortion_1"],
        intr["radial_distortion_2"],
        intr["radial_distortion_3"],
        intr["radial_distortion_4"],
    ], dtype=np.float64)

    w = data["image_width"]
    h = data["image_height"]
    reproj = data.get("camera_reproj_error", "?")

    return K, D, w, h, reproj


def main():
    import cv2

    # Calibration file path
    default_calib = "../universal_manipulation_interface/example/calibration/rpi_camera_intrinsics.json"
    calib_path = sys.argv[1] if len(sys.argv) > 1 else default_calib

    print(f"Loading calibration from: {calib_path}")
    K, D, cal_w, cal_h, reproj = load_calibration(calib_path)

    print(f"  Image size: {cal_w}x{cal_h}")
    print(f"  Focal length: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"  Principal point: ({K[0,2]:.1f}, {K[1,2]:.1f})")
    print(f"  Distortion: [{D[0]:.6f}, {D[1]:.6f}, {D[2]:.6f}, {D[3]:.6f}]")
    print(f"  Reproj error: {reproj}")
    print()

    # Capture a frame
    print("Capturing frame...")
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (cal_w, cal_h), "format": "RGB888"},
        )
        cam.configure(config)
        cam.start()
        time.sleep(1.0)  # let auto-exposure settle
        frame = cam.capture_array("main")
        cam.stop()
        cam.close()
    except ImportError:
        print("picamera2 not available — using a test pattern")
        frame = np.zeros((cal_h, cal_w, 3), dtype=np.uint8)
        # Draw a grid as test pattern
        for y in range(0, cal_h, 50):
            frame[y, :] = 128
        for x in range(0, cal_w, 50):
            frame[:, x] = 128

    h, w = frame.shape[:2]
    print(f"  Captured: {w}x{h}")

    if w != cal_w or h != cal_h:
        print(f"  WARNING: captured size {w}x{h} != calibration size {cal_w}x{cal_h}")

    # Save raw
    # picamera2 RGB888 is actually BGR from the ISP
    raw_bgr = frame  # already BGR for OpenCV
    cv2.imwrite("calibration_raw.jpg", raw_bgr)
    print("  Saved calibration_raw.jpg")

    # Undistort using fisheye model
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.5,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2,
    )
    undistorted = cv2.remap(raw_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Draw grid overlay on undistorted image to check straightness
    overlay = undistorted.copy()
    grid_color = (0, 255, 0)
    for y in range(0, h, h // 8):
        cv2.line(overlay, (0, y), (w, y), grid_color, 1)
    for x in range(0, w, w // 8):
        cv2.line(overlay, (x, 0), (x, h), grid_color, 1)
    # Mark principal point
    cx_new = int(new_K[0, 2])
    cy_new = int(new_K[1, 2])
    cv2.drawMarker(overlay, (cx_new, cy_new), (0, 0, 255),
                   cv2.MARKER_CROSS, 30, 2)

    cv2.imwrite("calibration_undistort.jpg", overlay)
    print("  Saved calibration_undistort.jpg")

    print()
    print("Check the images:")
    print("  - calibration_raw.jpg: fisheye distortion visible (curved lines)")
    print("  - calibration_undistort.jpg: lines should be straight if calibration is correct")
    print("  - Red cross marks the principal point")


if __name__ == "__main__":
    main()
