"""ArUco marker exploration for casquette FOV / size sizing.

Polls the casquette's /api/camera/snapshot endpoint, runs ArUco detection,
shows an OpenCV window with bounding boxes + IDs, and prints per-detection
pixel size and (optionally) a rough distance estimate. Use this to figure
out which physical marker size gives reliable detection at your target
operating distance from the head-mounted casquette to the grabettes.

Runs on the WORKSTATION (no opencv dependency on the Pi). Requires:
    pip install opencv-contrib-python numpy requests

(opencv-contrib-python ships the ArUco module; opencv-python alone may not.)

Generate test markers from the same dictionary the script uses, e.g. via:
    python -c "import cv2; cv2.imwrite('aruco_4x4_50_id0.png', \
        cv2.aruco.generateImageMarker(\
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 600))"
Print them at known physical sizes (3-6 cm typical), tape to a grabette body,
and observe.

Usage:
    python scripts/aruco_explore.py
    python scripts/aruco_explore.py --size-mm 40 --focal-px 850
    python scripts/aruco_explore.py --api http://casquette.local:8001
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
import requests

DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def fetch_snapshot(api_url: str) -> np.ndarray | None:
    try:
        r = requests.get(f"{api_url}/api/camera/snapshot", timeout=2.0)
        r.raise_for_status()
    except Exception as e:
        print(f"  [snapshot error] {e}", file=sys.stderr)
        return None
    arr = np.frombuffer(r.content, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def pinhole_distance_mm(marker_size_mm: float, marker_px: float, focal_px: float) -> float:
    """Pinhole approximation: d = (S * f) / s_px.

    Inaccurate near the fisheye image edges where distortion compresses
    apparent marker size. Use only for ballpark distance feedback during
    FOV experimentation; for real pose work, solvePnP with calibrated
    intrinsics + distortion coefficients.
    """
    if marker_px <= 0:
        return float("inf")
    return (marker_size_mm * focal_px) / marker_px


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", default="http://casquette.local:8001",
                   help="Base URL of the casquette daemon")
    p.add_argument("--size-mm", type=float, default=50.0,
                   help="Physical size of the printed marker, mm "
                        "(used only for the rough distance estimate)")
    p.add_argument("--dict", default="DICT_4X4_50", choices=list(DICT_MAP),
                   help="ArUco dictionary to detect against")
    p.add_argument("--rate-hz", type=float, default=5.0,
                   help="Snapshot poll rate; higher = smoother, more network load")
    p.add_argument("--focal-px", type=float, default=850.0,
                   help="Approximate focal length in pixels for the rough "
                        "distance estimate. Grabette V1 (OV5647 + fisheye) "
                        "calibration gives ~850 px at 1296x972 — adjust if "
                        "casquette differs")
    args = p.parse_args()

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dict]),
        cv2.aruco.DetectorParameters(),
    )
    dt = 1.0 / args.rate_hz

    print(f"Polling {args.api}/api/camera/snapshot at {args.rate_hz:.1f} Hz")
    print(f"Detecting {args.dict}, assumed marker size {args.size_mm:.1f} mm, "
          f"focal {args.focal_px:.0f} px")
    print(f"Press 'q' in the window to quit.\n")

    last_print = 0.0
    while True:
        t0 = time.monotonic()
        img = fetch_snapshot(args.api)
        if img is None:
            time.sleep(dt)
            continue

        corners, ids, _rejected = detector.detectMarkers(img)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            # One log line per marker per snapshot — rate-limit to avoid
            # flooding the terminal.
            do_log = (time.monotonic() - last_print) > 0.2
            if do_log:
                last_print = time.monotonic()
            for i, marker_corners in enumerate(corners):
                pts = marker_corners[0]  # (4, 2)
                sides = [np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]
                side_px = float(np.mean(sides))
                d_cm = pinhole_distance_mm(args.size_mm, side_px, args.focal_px) / 10.0
                mid = pts.mean(axis=0).astype(int)
                label = f"id={int(ids[i][0])} {side_px:.0f}px ~{d_cm:.0f}cm"
                cv2.putText(img, label, tuple(mid),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if do_log:
                    print(f"  id={int(ids[i][0]):3d}  side={side_px:5.0f}px  "
                          f"est_dist≈{d_cm:5.1f}cm")
        else:
            # Light feedback so the user knows we're actually polling.
            print("  no markers", end="\r")

        cv2.imshow("Casquette ArUco Explore", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        sleep = dt - (time.monotonic() - t0)
        if sleep > 0:
            time.sleep(sleep)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
