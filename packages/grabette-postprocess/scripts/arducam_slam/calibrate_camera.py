#!/usr/bin/env python3
"""
Calibrate camera intrinsics from a ChArUco video and update SLAM settings.

Performs OpenCV fisheye calibration, saves intrinsics JSON, and updates the
SLAM settings YAML with the new intrinsics. The IMU parameters and extrinsics
(T_b_c1) are left unchanged — only camera intrinsics are updated.

Usage:
    uv run python scripts/calibrate_camera.py --video /path/to/charuco_video.mp4

    # Dry run (calibrate only, don't update YAML)
    uv run python scripts/calibrate_camera.py --video /path/to/charuco_video.mp4 --dry-run

    # Custom output paths
    uv run python scripts/calibrate_camera.py --video /path/to/charuco_video.mp4 \
        --output intrinsics.json \
        --settings config/rpi_bmi088_slam_settings.yaml
"""

import json
import re
from pathlib import Path

import click
import cv2
import cv2.aruco as aruco
import numpy as np


_PACKAGE_DIR = Path(__file__).parent.parent
_DEFAULT_SETTINGS = _PACKAGE_DIR / "config" / "rpi_bmi088_slam_settings.yaml"
_DEFAULT_OUTPUT = _PACKAGE_DIR / "config" / "camera_intrinsics.json"


def detect_best_dictionary(video_path: str, num_test_frames: int = 5) -> str | None:
    """Auto-detect the ArUco dictionary used in the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / (num_test_frames + 1))
                     for i in range(1, num_test_frames + 1)]

    candidates = [
        ("DICT_ARUCO_ORIGINAL", aruco.DICT_ARUCO_ORIGINAL),
        ("DICT_4X4_250", aruco.DICT_4X4_250),
        ("DICT_5X5_250", aruco.DICT_5X5_250),
        ("DICT_6X6_250", aruco.DICT_6X6_250),
    ]
    results = {name: 0 for name, _ in candidates}

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for dict_name, dict_id in candidates:
            dictionary = aruco.getPredefinedDictionary(dict_id)
            detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
            _, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                results[dict_name] += len(ids)

    cap.release()
    best_name, best_count = max(results.items(), key=lambda x: x[1])
    return best_name if best_count > 0 else None


def calibrate_fisheye(
    video_path: str,
    square_size: float = 0.019,
    cols: int = 10,
    rows: int = 8,
    dictionary_name: str = "DICT_ARUCO_ORIGINAL",
    min_corners: int = 6,
    max_frames: int = 100,
    voxel_size: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Calibrate fisheye camera from ChArUco video.

    Returns (K, D, width, height, rms_error).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise click.ClickException(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    # Create ChArUco board
    dict_id = getattr(aruco, dictionary_name)
    dictionary = aruco.getPredefinedDictionary(dict_id)
    marker_size = square_size * 0.5
    board = aruco.CharucoBoard((cols, rows), square_size, marker_size, dictionary)
    # Our boards were printed before OpenCV 4.8 changed the default layout
    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)

    charuco_detector = aruco.CharucoDetector(board)

    # Collect calibration corners with voxel grid filtering
    all_obj_points = []
    all_img_points = []
    used_positions = set()
    board_corners_3d = board.getChessboardCorners()

    print(f"\nExtracting ChArUco corners (dictionary: {dictionary_name})...")
    frame_idx = 0
    n_views = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 10 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)

        if charuco_corners is None or len(charuco_corners) < min_corners:
            continue

        # Voxel filtering to avoid redundant views
        mean_pos = charuco_corners.mean(axis=0)[0]
        voxel_key = (
            int(mean_pos[0] / (width * voxel_size)),
            int(mean_pos[1] / (height * voxel_size)),
        )
        if voxel_key in used_positions:
            continue
        used_positions.add(voxel_key)

        # Build object/image point arrays
        obj_pts = []
        img_pts = []
        for corner, cid in zip(charuco_corners, charuco_ids):
            cid = cid[0]
            if 0 <= cid < len(board_corners_3d):
                obj_pts.append(board_corners_3d[cid])
                img_pts.append(corner[0])

        if len(obj_pts) < min_corners:
            continue

        all_obj_points.append(np.array(obj_pts, dtype=np.float32).reshape(-1, 1, 3))
        all_img_points.append(np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2))
        n_views += 1
        print(f"  Frame {frame_idx}: {len(obj_pts)} corners (views: {n_views})")

        if n_views >= max_frames:
            break

    cap.release()

    if n_views < 10:
        raise click.ClickException(
            f"Only {n_views} views collected (need >= 10). "
            "Make sure the board is visible and well-lit."
        )

    print(f"\nCollected {n_views} calibration views")

    # Initial pinhole estimate for fisheye initialization
    print("Running initial pinhole calibration...")
    try:
        _, K_init, _, _, _ = cv2.calibrateCamera(
            all_obj_points, all_img_points, (width, height), None, None,
            flags=cv2.CALIB_FIX_K3,
        )
    except cv2.error:
        K_init = np.eye(3)
        K_init[0, 0] = K_init[1, 1] = min(width, height) * 0.8
        K_init[0, 2] = width / 2
        K_init[1, 2] = height / 2

    # Fisheye calibration with retry on ill-conditioned views
    print("Running fisheye calibration...")
    K = K_init.copy()
    D = np.zeros((4, 1))
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_FIX_SKEW
        + cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    )
    current_obj = all_obj_points.copy()
    current_img = all_img_points.copy()

    for attempt in range(5):
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                current_obj, current_img, (width, height),
                K.copy(), D.copy(), flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            )
            break
        except cv2.error as e:
            if "Ill-conditioned" in str(e) and len(current_obj) > 15:
                match = re.search(r'input array (\d+)', str(e))
                if match:
                    bad_idx = int(match.group(1))
                    print(f"  Removing ill-conditioned view {bad_idx} (attempt {attempt + 1})")
                    current_obj = [p for i, p in enumerate(current_obj) if i != bad_idx]
                    current_img = [p for i, p in enumerate(current_img) if i != bad_idx]
                    continue
            raise
    else:
        raise click.ClickException("Calibration failed after 5 attempts")

    print(f"\n{'=' * 50}")
    print(f"  RMS reprojection error: {rms:.4f} pixels")
    print(f"  fx={K[0,0]:.4f}  fy={K[1,1]:.4f}")
    print(f"  cx={K[0,2]:.4f}  cy={K[1,2]:.4f}")
    print(f"  k1={D[0,0]:.6f}  k2={D[1,0]:.6f}  k3={D[2,0]:.6f}  k4={D[3,0]:.6f}")
    print(f"{'=' * 50}")

    if rms < 0.5:
        print("  Quality: EXCELLENT")
    elif rms < 1.0:
        print("  Quality: GOOD")
    else:
        print("  Quality: POOR — consider recalibrating")

    return K, D, width, height, rms


def save_intrinsics_json(path: Path, K, D, width, height, rms, fps=49.62):
    """Save calibration in UMI-compatible JSON format."""
    data = {
        "image_width": width,
        "image_height": height,
        "intrinsic_type": "FISHEYE",
        "intrinsics": {
            "aspect_ratio": float(K[1, 1] / K[0, 0]),
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
        "camera_reproj_error": float(rms),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print(f"\nSaved intrinsics to: {path}")


def update_slam_settings(
    settings_path: Path,
    K: np.ndarray,
    D: np.ndarray,
    calib_width: int,
    calib_height: int,
    rms: float,
):
    """Update camera intrinsics in the SLAM settings YAML.

    If the YAML target resolution differs from the calibration resolution,
    intrinsics are scaled accordingly. Only camera intrinsic lines are
    modified; everything else (IMU, ORB, extrinsics) is preserved.
    """
    text = settings_path.read_text()

    # Read target resolution from YAML
    m_w = re.search(r'^Camera\.width:\s*(\d+)', text, re.MULTILINE)
    m_h = re.search(r'^Camera\.height:\s*(\d+)', text, re.MULTILINE)
    if not m_w or not m_h:
        raise click.ClickException(f"Cannot find Camera.width/height in {settings_path}")
    target_w = int(m_w.group(1))
    target_h = int(m_h.group(1))

    # Scale intrinsics if target resolution differs from calibration resolution
    scale_x = target_w / calib_width
    scale_y = target_h / calib_height
    fx = K[0, 0] * scale_x
    fy = K[1, 1] * scale_y
    cx = K[0, 2] * scale_x
    cy = K[1, 2] * scale_y

    if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
        print(f"\nScaling intrinsics from {calib_width}x{calib_height} "
              f"to {target_w}x{target_h} (scale {scale_x:.4f}x{scale_y:.4f})")

    # Replace intrinsic values in YAML (preserve formatting)
    replacements = {
        r'(Camera1\.fx:\s*)[\d.e+-]+': f'\\g<1>{fx}',
        r'(Camera1\.fy:\s*)[\d.e+-]+': f'\\g<1>{fy}',
        r'(Camera1\.cx:\s*)[\d.e+-]+': f'\\g<1>{cx}',
        r'(Camera1\.cy:\s*)[\d.e+-]+': f'\\g<1>{cy}',
        r'(Camera1\.k1:\s*)[\d.e+-]+': f'\\g<1>{D[0,0]}',
        r'(Camera1\.k2:\s*)[\d.e+-]+': f'\\g<1>{D[1,0]}',
        r'(Camera1\.k3:\s*)[\d.e+-]+': f'\\g<1>{D[2,0]}',
        r'(Camera1\.k4:\s*)[\d.e+-]+': f'\\g<1>{D[3,0]}',
    }

    for pattern, replacement in replacements.items():
        text, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if n == 0:
            param = pattern.split(r'\.')[1].split(r':')[0]
            print(f"  Warning: {param} not found in YAML")

    # Update the intrinsics comment
    text = re.sub(
        r'(# Camera 1 intrinsics).*',
        f'\\g<1> at {target_w}x{target_h} (reproj error: {rms:.2f}px)',
        text, count=1,
    )
    text = re.sub(
        r'# Scaled from.*\n',
        '',
        text, count=1,
    )

    settings_path.write_text(text)
    print(f"Updated SLAM settings: {settings_path}")
    print(f"  fx={fx:.4f}  fy={fy:.4f}  cx={cx:.4f}  cy={cy:.4f}")


@click.command()
@click.option("--video", "-v", required=True, type=click.Path(exists=True),
              help="Path to ChArUco calibration video")
@click.option("--output", "-o", default=str(_DEFAULT_OUTPUT), type=click.Path(),
              help="Output intrinsics JSON path")
@click.option("--settings", "-s", default=str(_DEFAULT_SETTINGS), type=click.Path(exists=True),
              help="SLAM settings YAML to update")
@click.option("--square_size", type=float, default=0.019,
              help="ChArUco square size in meters")
@click.option("--cols", type=int, default=10, help="Board columns")
@click.option("--rows", type=int, default=8, help="Board rows")
@click.option("--dictionary", "-d", default=None, help="ArUco dictionary (auto-detected)")
@click.option("--voxel_size", type=float, default=0.1,
              help="Voxel grid size for view filtering (0-1)")
@click.option("--max_frames", type=int, default=100, help="Max calibration frames")
@click.option("--dry-run", is_flag=True, default=False,
              help="Calibrate only, don't update SLAM settings")
def main(video, output, settings, square_size, cols, rows, dictionary,
         voxel_size, max_frames, dry_run):
    """Calibrate camera and update SLAM settings."""

    # Auto-detect dictionary
    if dictionary is None:
        print("Auto-detecting ArUco dictionary...")
        dictionary = detect_best_dictionary(video)
        if dictionary is None:
            raise click.ClickException("No ArUco markers detected. Use --dictionary.")
        print(f"  Detected: {dictionary}")

    # Calibrate
    K, D, width, height, rms = calibrate_fisheye(
        video, square_size, cols, rows, dictionary,
        max_frames=max_frames, voxel_size=voxel_size,
    )

    # Save intrinsics JSON
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    save_intrinsics_json(Path(output), K, D, width, height, rms, fps)

    # Update SLAM settings
    if not dry_run:
        update_slam_settings(Path(settings), K, D, width, height, rms)
    else:
        print("\nDry run — SLAM settings not updated")


if __name__ == "__main__":
    main()
