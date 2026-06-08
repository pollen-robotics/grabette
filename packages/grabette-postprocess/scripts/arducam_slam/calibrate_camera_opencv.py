#!/usr/bin/env python3
"""
Camera calibration using OpenCV with configurable ArUco dictionary.

This script performs camera intrinsic calibration using a ChArUco board.
It supports any ArUco dictionary and outputs UMI-compatible intrinsics JSON.
"""

import argparse
import json
import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict


# ArUco dictionary mapping
ARUCO_DICTIONARIES = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_4X4_1000": aruco.DICT_4X4_1000,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_5X5_1000": aruco.DICT_5X5_1000,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_6X6_1000": aruco.DICT_6X6_1000,
    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_7X7_250": aruco.DICT_7X7_250,
    "DICT_7X7_1000": aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
}


def detect_best_dictionary(video_path: str, num_test_frames: int = 5) -> Optional[str]:
    """Auto-detect the ArUco dictionary used in the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / (num_test_frames + 1)) for i in range(1, num_test_frames + 1)]

    results = {name: 0 for name in ARUCO_DICTIONARIES}

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for dict_name, dict_id in ARUCO_DICTIONARIES.items():
            dictionary = aruco.getPredefinedDictionary(dict_id)
            detector_params = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(dictionary, detector_params)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                results[dict_name] += len(ids)

    cap.release()

    best = max(results.items(), key=lambda x: x[1])
    if best[1] > 0:
        return best[0]
    return None


def calibrate_fisheye_camera(
    video_path: str,
    square_size: float,
    cols: int,
    rows: int,
    dictionary_name: str,
    min_corners: int = 6,
    max_frames: int = 100,
    voxel_size: float = 0.1,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int, float, List[Dict]]:
    """
    Calibrate fisheye camera using ChArUco board.

    Returns:
        K: Camera matrix (3x3)
        D: Distortion coefficients (4,)
        width: Image width
        height: Image height
        rms: RMS reprojection error
        views: List of calibration view info
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    # Create ChArUco board
    dict_id = ARUCO_DICTIONARIES[dictionary_name]
    dictionary = aruco.getPredefinedDictionary(dict_id)
    marker_size = square_size * 0.5  # Markers are half the square size
    board = aruco.CharucoBoard((cols, rows), square_size, marker_size, dictionary)

    # Detection parameters
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)
    charuco_detector = aruco.CharucoDetector(board)

    # Collect calibration corners
    all_corners = []
    all_ids = []
    image_size = (width, height)

    # For voxel grid filtering (avoid redundant views)
    used_positions = set()

    frame_idx = 0
    views = []

    print(f"\nExtracting ChArUco corners (dictionary: {dictionary_name})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 10 != 0:  # Process every 10th frame
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ChArUco corners
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

        if charuco_corners is not None and len(charuco_corners) >= min_corners:
            # Compute mean position for voxel filtering
            mean_pos = charuco_corners.mean(axis=0)[0]
            voxel_key = (
                int(mean_pos[0] / (width * voxel_size)),
                int(mean_pos[1] / (height * voxel_size)),
            )

            if voxel_key not in used_positions:
                used_positions.add(voxel_key)
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                views.append({
                    'frame': frame_idx,
                    'corners': len(charuco_corners),
                    'position': mean_pos.tolist(),
                })

                print(f"  Frame {frame_idx}: {len(charuco_corners)} corners (total views: {len(all_corners)})")

                if len(all_corners) >= max_frames:
                    print(f"  Reached max frames ({max_frames})")
                    break

    cap.release()

    if len(all_corners) < 10:
        print(f"[ERROR] Only {len(all_corners)} views collected. Need at least 10 for reliable calibration.")
        return None, None, width, height, float('inf'), views

    print(f"\nCollected {len(all_corners)} calibration views")

    # Prepare object points - use board corner positions directly
    obj_points = []
    img_points = []

    # Get board corner positions (indexed by corner ID)
    board_corners_3d = board.getChessboardCorners()

    for corners, ids in zip(all_corners, all_ids):
        if ids is None or len(ids) < min_corners:
            continue

        # Build object and image point arrays manually
        obj_pts = []
        img_pts = []

        for corner, cid in zip(corners, ids):
            cid = cid[0]
            if 0 <= cid < len(board_corners_3d):
                obj_pts.append(board_corners_3d[cid])
                img_pts.append(corner[0])

        if len(obj_pts) >= min_corners:
            obj_points.append(np.array(obj_pts, dtype=np.float32).reshape(-1, 1, 3))
            img_points.append(np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2))

    print(f"Valid views for calibration: {len(obj_points)}")

    if len(obj_points) < 10:
        print("[ERROR] Not enough valid views for calibration")
        return None, None, width, height, float('inf'), views

    # First run standard calibration to get initial estimates
    print("\nRunning initial pinhole calibration for estimates...")

    # Convert points to format expected by calibrateCamera
    obj_pts_std = [pts.astype(np.float32) for pts in obj_points]
    img_pts_std = [pts.astype(np.float32) for pts in img_points]

    try:
        rms_pinhole, K_init, dist_pinhole, _, _ = cv2.calibrateCamera(
            obj_pts_std,
            img_pts_std,
            image_size,
            None,
            None,
            flags=cv2.CALIB_FIX_K3
        )
        print(f"  Pinhole calibration RMS: {rms_pinhole:.4f}")
        print(f"  Initial focal length: {K_init[0,0]:.2f}")
    except cv2.error as e:
        print(f"  Pinhole calibration failed: {e}")
        K_init = np.eye(3)
        K_init[0, 0] = K_init[1, 1] = min(width, height) * 0.8
        K_init[0, 2] = width / 2
        K_init[1, 2] = height / 2

    # Fisheye calibration
    print("\nRunning fisheye calibration...")

    # Use pinhole estimates as initialization
    K = K_init.copy()
    D = np.zeros((4, 1))

    # Calibration flags - start without CHECK_COND, add if stable
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_FIX_SKEW +
        cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    )

    # Try calibration, removing problematic views if needed
    max_attempts = 5
    current_obj_points = obj_points.copy()
    current_img_points = img_points.copy()

    for attempt in range(max_attempts):
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                current_obj_points,
                current_img_points,
                image_size,
                K.copy(),
                D.copy(),
                flags=calibration_flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            )
            break  # Success
        except cv2.error as e:
            if "Ill-conditioned" in str(e) and len(current_obj_points) > 15:
                # Remove the problematic view mentioned in error
                import re
                match = re.search(r'input array (\d+)', str(e))
                if match:
                    bad_idx = int(match.group(1))
                    print(f"  Removing ill-conditioned view {bad_idx} (attempt {attempt + 1})")
                    current_obj_points = [p for i, p in enumerate(current_obj_points) if i != bad_idx]
                    current_img_points = [p for i, p in enumerate(current_img_points) if i != bad_idx]
                    continue
            raise  # Re-raise if can't handle
    else:
        # Loop completed without success
        print(f"[ERROR] Calibration failed after {max_attempts} attempts")
        return None, None, width, height, float('inf'), views

    print(f"\n{'=' * 50}")
    print("Calibration Results")
    print(f"{'=' * 50}")
    print(f"RMS reprojection error: {rms:.4f} pixels")
    print(f"Camera matrix K:")
    print(f"  fx = {K[0,0]:.2f}")
    print(f"  fy = {K[1,1]:.2f}")
    print(f"  cx = {K[0,2]:.2f}")
    print(f"  cy = {K[1,2]:.2f}")
    print(f"Distortion D (k1-k4):")
    print(f"  k1 = {D[0,0]:.6f}")
    print(f"  k2 = {D[1,0]:.6f}")
    print(f"  k3 = {D[2,0]:.6f}")
    print(f"  k4 = {D[3,0]:.6f}")

    return K, D, width, height, rms, views


def save_intrinsics_json(
    output_path: str,
    K: np.ndarray,
    D: np.ndarray,
    width: int,
    height: int,
    fps: float = 49.62,
    t_i_c: Optional[np.ndarray] = None,
    q_i_c: Optional[np.ndarray] = None,
):
    """Save calibration in UMI-compatible JSON format."""
    intrinsics = {
        "image_width": width,
        "image_height": height,
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": K[1, 1] / K[0, 0],
            "focal_length": float(K[0, 0]),
            "principal_pt_x": float(K[0, 2]),
            "principal_pt_y": float(K[1, 2]),
            "radial_distortion_1": float(D[0, 0]),
            "radial_distortion_2": float(D[1, 0]),
            "radial_distortion_3": float(D[2, 0]),
            "radial_distortion_4": float(D[3, 0]),
            "skew": 0.0,
        },
        "fps": fps,
    }

    # Add IMU-camera transform if provided
    if t_i_c is not None and q_i_c is not None:
        intrinsics["t_i_c"] = {"x": float(t_i_c[0]), "y": float(t_i_c[1]), "z": float(t_i_c[2])}
        intrinsics["q_i_c"] = {"w": float(q_i_c[0]), "x": float(q_i_c[1]), "y": float(q_i_c[2]), "z": float(q_i_c[3])}
    else:
        # Identity transform placeholder
        intrinsics["t_i_c"] = {"x": 0.0, "y": 0.0, "z": 0.0}
        intrinsics["q_i_c"] = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(intrinsics, f, indent=2)

    print(f"\nSaved intrinsics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Camera calibration using ChArUco board with configurable ArUco dictionary"
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to calibration video"
    )
    parser.add_argument(
        "--output", "-o",
        default="camera_intrinsics.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--square_size", "-s",
        type=float,
        default=0.019,
        help="ChArUco square size in meters (default: 0.019)"
    )
    parser.add_argument(
        "--cols", "-c",
        type=int,
        default=10,
        help="Number of columns in ChArUco board (default: 10)"
    )
    parser.add_argument(
        "--rows", "-r",
        type=int,
        default=8,
        help="Number of rows in ChArUco board (default: 8)"
    )
    parser.add_argument(
        "--dictionary", "-d",
        choices=list(ARUCO_DICTIONARIES.keys()),
        default=None,
        help="ArUco dictionary (auto-detected if not specified)"
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.1,
        help="Voxel grid size for view filtering (0-1, default: 0.1)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Maximum calibration frames to use (default: 100)"
    )
    parser.add_argument(
        "--min_corners",
        type=int,
        default=6,
        help="Minimum corners per frame (default: 6)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=49.62,
        help="Camera FPS for intrinsics file (default: 49.62)"
    )

    args = parser.parse_args()

    # Auto-detect dictionary if not specified
    dictionary = args.dictionary
    if dictionary is None:
        print("Auto-detecting ArUco dictionary...")
        dictionary = detect_best_dictionary(args.video)
        if dictionary is None:
            print("[ERROR] Could not detect any ArUco markers in video!")
            print("Please specify --dictionary manually or check that the board is visible.")
            return 1
        print(f"Detected dictionary: {dictionary}")

    # Run calibration
    K, D, width, height, rms, views = calibrate_fisheye_camera(
        video_path=args.video,
        square_size=args.square_size,
        cols=args.cols,
        rows=args.rows,
        dictionary_name=dictionary,
        min_corners=args.min_corners,
        max_frames=args.max_frames,
        voxel_size=args.voxel_size,
    )

    if K is None:
        print("\n[ERROR] Calibration failed!")
        return 1

    # Assess quality
    if rms < 0.5:
        quality = "EXCELLENT"
    elif rms < 1.0:
        quality = "GOOD"
    else:
        quality = "POOR - consider recalibrating"

    print(f"\nCalibration quality: {quality}")

    # Save results
    save_intrinsics_json(
        output_path=args.output,
        K=K,
        D=D,
        width=width,
        height=height,
        fps=args.fps,
    )

    return 0


if __name__ == "__main__":
    exit(main())
