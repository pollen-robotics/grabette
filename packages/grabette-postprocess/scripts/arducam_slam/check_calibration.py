#!/usr/bin/env python3
"""
Quick calibration check: detect ChArUco corners in images/video from a device
and compute reprojection error using existing intrinsics.

If median reprojection error < 1.0px: intrinsics are good, no need to recalibrate.
If median reprojection error > 2.0px: consider recalibrating for this device.

Baseline on the same device: median ~0.5px, mean ~0.7px.

Usage:
    # From video (samples N frames automatically)
    uv run python scripts/check_calibration.py --video /path/to/video.mp4

    # From images
    uv run python scripts/check_calibration.py --images img1.png img2.png img3.png

    # Custom intrinsics file
    uv run python scripts/check_calibration.py --video /path/to/video.mp4 \
        --intrinsics /path/to/intrinsics.json
"""

import json

import click
import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path


# Default intrinsics shipped with the parent GRABETTE project
_DEFAULT_INTRINSICS = (
    Path(__file__).parent.parent.parent
    / "universal_manipulation_interface"
    / "example"
    / "calibration"
    / "rpi_camera_intrinsics.json"
)


def load_intrinsics(path: str):
    """Load intrinsics from UMI-compatible JSON."""
    with open(path) as f:
        data = json.load(f)

    intr = data["intrinsics"]
    fx = intr["focal_length"]
    fy = fx * intr["aspect_ratio"]
    cx = intr["principal_pt_x"]
    cy = intr["principal_pt_y"]

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array([
        intr["radial_distortion_1"],
        intr["radial_distortion_2"],
        intr["radial_distortion_3"],
        intr["radial_distortion_4"],
    ], dtype=np.float64).reshape(4, 1)

    return K, D, data["image_width"], data["image_height"]


def compute_reprojection_error(
    gray: np.ndarray,
    board: aruco.CharucoBoard,
    charuco_detector: aruco.CharucoDetector,
    K: np.ndarray,
    D: np.ndarray,
    min_corners: int = 6,
):
    """Detect ChArUco corners and compute fisheye reprojection error.

    Returns (n_corners, rms_error) or (0, None) if detection fails.
    """
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

    if charuco_corners is None or len(charuco_corners) < min_corners:
        return 0, None

    # Build object/image point arrays
    board_corners_3d = board.getChessboardCorners()
    obj_pts = []
    img_pts = []
    for corner, cid in zip(charuco_corners, charuco_ids):
        cid = cid[0]
        if 0 <= cid < len(board_corners_3d):
            obj_pts.append(board_corners_3d[cid])
            img_pts.append(corner[0])

    if len(obj_pts) < min_corners:
        return 0, None

    obj_pts = np.array(obj_pts, dtype=np.float64)
    img_pts = np.array(img_pts, dtype=np.float64)

    # Undistort detected corners to normalized camera coordinates using fisheye model,
    # then solvePnP in normalized space (identity K, no distortion).
    # This avoids solvePnP using the wrong (pinhole) distortion model.
    undistorted = cv2.fisheye.undistortPoints(
        img_pts.reshape(-1, 1, 2), K, D
    ).reshape(-1, 2)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, undistorted, np.eye(3, dtype=np.float64), None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return len(obj_pts), None

    # Reproject through full fisheye model (K + D) and compare with original detections
    projected, _ = cv2.fisheye.projectPoints(
        obj_pts.reshape(-1, 1, 3), rvec, tvec, K, D
    )
    projected = projected.reshape(-1, 2)

    errors = np.linalg.norm(img_pts - projected, axis=1)
    rms = np.sqrt(np.mean(errors ** 2))

    return len(obj_pts), rms


def detect_dictionary(gray: np.ndarray):
    """Try common ArUco dictionaries on a frame, return the best one."""
    best_name = None
    best_count = 0
    for name, dict_id in [
        ("DICT_ARUCO_ORIGINAL", aruco.DICT_ARUCO_ORIGINAL),
        ("DICT_4X4_250", aruco.DICT_4X4_250),
        ("DICT_5X5_250", aruco.DICT_5X5_250),
        ("DICT_6X6_250", aruco.DICT_6X6_250),
        ("DICT_4X4_50", aruco.DICT_4X4_50),
        ("DICT_5X5_50", aruco.DICT_5X5_50),
        ("DICT_6X6_50", aruco.DICT_6X6_50),
    ]:
        d = aruco.getPredefinedDictionary(dict_id)
        detector = aruco.ArucoDetector(d, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
        n = len(ids) if ids is not None else 0
        if n > best_count:
            best_count = n
            best_name = name
    return best_name, best_count


def sample_frames_from_video(video_path: str, n_frames: int = 10):
    """Sample N evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Evenly spaced, skip first/last 10%
    start = int(total * 0.1)
    end = int(total * 0.9)
    indices = np.linspace(start, end, n_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

    cap.release()
    return frames, width, height


@click.command()
@click.option("--video", "-v", type=click.Path(exists=True), help="Path to video file")
@click.option("--images", "-i", multiple=True, type=click.Path(exists=True), help="Path to image files")
@click.option("--intrinsics", default=str(_DEFAULT_INTRINSICS), type=click.Path(exists=True),
              help="Path to intrinsics JSON")
@click.option("--square_size", type=float, default=0.019, help="ChArUco square size in meters")
@click.option("--cols", type=int, default=10, help="Board columns")
@click.option("--rows", type=int, default=8, help="Board rows")
@click.option("--dictionary", "-d", default=None, help="ArUco dictionary (auto-detected if not specified)")
@click.option("--n_frames", type=int, default=20, help="Frames to sample from video")
def main(video, images, intrinsics, square_size, cols, rows, dictionary, n_frames):
    """Check if existing camera intrinsics fit a new device."""
    if not video and not images:
        raise click.UsageError("Provide either --video or --images")

    # Load intrinsics
    print(f"Loading intrinsics from: {intrinsics}")
    K, D, calib_w, calib_h = load_intrinsics(intrinsics)
    print(f"  Calibrated for: {calib_w}x{calib_h}")
    print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"  D=[{D[0,0]:.6f}, {D[1,0]:.6f}, {D[2,0]:.6f}, {D[3,0]:.6f}]")

    # Load frames
    if video:
        print(f"\nSampling {n_frames} frames from: {video}")
        frames, vid_w, vid_h = sample_frames_from_video(video, n_frames)
        print(f"  Video resolution: {vid_w}x{vid_h}")
        if vid_w != calib_w or vid_h != calib_h:
            print(f"  WARNING: resolution mismatch! Intrinsics were calibrated for {calib_w}x{calib_h}")
    else:
        frames = []
        for p in images:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  WARNING: cannot read {p}, skipping")
                continue
            frames.append((p, img))
        if not frames:
            raise click.ClickException("No valid images")

    # Auto-detect or use specified dictionary
    dict_name = dictionary
    if dict_name is None:
        print("\nAuto-detecting ArUco dictionary...")
        _, test_gray = frames[len(frames) // 2]
        dict_name, n_detected = detect_dictionary(test_gray)
        if dict_name is None or n_detected == 0:
            for _, g in frames[::max(1, len(frames) // 5)]:
                dict_name, n_detected = detect_dictionary(g)
                if n_detected > 0:
                    break
        if dict_name and n_detected > 0:
            print(f"  Detected: {dict_name} ({n_detected} markers)")
        else:
            print("  WARNING: no ArUco markers detected in any frame!")
            print("  Defaulting to DICT_ARUCO_ORIGINAL. Use --dictionary to override.")
            dict_name = "DICT_ARUCO_ORIGINAL"
    else:
        print(f"\nUsing dictionary: {dict_name}")

    # Setup ChArUco detector
    dict_id = getattr(aruco, dict_name)
    aruco_dict = aruco.getPredefinedDictionary(dict_id)
    marker_size = square_size * 0.5
    board = aruco.CharucoBoard((cols, rows), square_size, marker_size, aruco_dict)
    # Our boards were printed before OpenCV 4.8 changed the default layout
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    charuco_detector = aruco.CharucoDetector(board)

    # Check each frame
    print(f"\nChecking reprojection error on {len(frames)} frames...\n")
    print(f"  {'Frame':<12} {'Corners':<10} {'RMS (px)':<10}")
    print(f"  {'-'*32}")

    errors = []
    total_corners = 0
    for label, gray in frames:
        n_corners, rms = compute_reprojection_error(gray, board, charuco_detector, K, D)
        if rms is not None:
            errors.append(rms)
            total_corners += n_corners
            print(f"  {str(label):<12} {n_corners:<10} {rms:<10.4f}")
        else:
            print(f"  {str(label):<12} {'no board' if n_corners == 0 else 'PnP fail':<10} {chr(0x2014):<10}")

    if not errors:
        raise click.ClickException(
            "No ChArUco board detected in any frame! "
            "Make sure the board is visible and well-lit."
        )

    # Summary
    mean_rms = np.mean(errors)
    median_rms = np.median(errors)
    max_rms = np.max(errors)

    print(f"\n{'='*40}")
    print(f"  Frames with detections: {len(errors)}/{len(frames)}")
    print(f"  Total corners used:     {total_corners}")
    print(f"  Mean RMS error:         {mean_rms:.4f} px")
    print(f"  Median RMS error:       {median_rms:.4f} px")
    print(f"  Max RMS error:          {max_rms:.4f} px")
    print(f"{'='*40}")

    if median_rms < 1.0:
        print(f"\n  RESULT: GOOD — intrinsics fit this device (median {median_rms:.2f}px < 1.0px)")
        print(f"  No recalibration needed.")
    elif median_rms < 2.0:
        print(f"\n  RESULT: MARGINAL — intrinsics are usable but not ideal (median {median_rms:.2f}px)")
        print(f"  Recalibration recommended for best SLAM performance.")
    else:
        print(f"\n  RESULT: POOR — intrinsics don't fit this device (median {median_rms:.2f}px > 2.0px)")
        print(f"  Recalibration required.")


if __name__ == "__main__":
    main()
