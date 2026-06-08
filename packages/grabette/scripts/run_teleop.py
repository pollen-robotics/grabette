#!/usr/bin/env python3
"""Standalone CLI to exercise the OakdTeleop class (Phase 2.1 validation).

Runs OakdTeleop, prints framerate / pose stats, and optionally streams to
a rerun viewer for visual inspection.

Same SLAM behavior as `teleop_vslam_test.py --backend basalt` but uses the
new `grabette.hardware.oakd_teleop.OakdTeleop` class — verifies the
extracted class works correctly before we wire it into the daemon.

Usage:
    uv run python scripts/run_teleop.py --duration 30
    uv run python scripts/run_teleop.py --rerun-host 192.168.1.22:9876
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Allow running directly from the repo without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grabette.hardware.oakd_teleop import OakdTeleop  # noqa: E402


def _setup_rerun(host: str):
    """Best-effort connection to a rerun viewer. Returns the module or None."""
    try:
        import rerun as rr
    except ImportError:
        print("rerun-sdk not installed; skipping live viz", file=sys.stderr)
        return None
    rr.init("grabette_teleop_run", spawn=False)
    if hasattr(rr, "connect_grpc"):
        rr.connect_grpc(f"rerun+http://{host}/proxy")
    elif hasattr(rr, "connect_tcp"):
        rr.connect_tcp(host)
    elif hasattr(rr, "connect"):
        rr.connect(host)
    else:
        print(f"no usable rerun connect API on rerun {rr.__version__}", file=sys.stderr)
        return None
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/axes", rr.Arrows3D(
        origins=[[0, 0, 0]] * 3,
        vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    ), static=True)
    print(f"rerun: streaming to {host}")
    return rr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=None,
                    help="run for N seconds then stop (default: until Ctrl-C)")
    ap.add_argument("--rerun-host", default=None,
                    help="stream to a rerun viewer at host:port (e.g. 192.168.1.5:9876)")
    ap.add_argument("--print-every", type=float, default=1.0,
                    help="print stats every N seconds (default 1.0; 0 = silent)")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    rr = _setup_rerun(args.rerun_host) if args.rerun_host else None

    teleop = OakdTeleop()
    teleop.init_device()
    teleop.start()
    print("OakdTeleop running. Ctrl-C to stop.")

    t0 = time.monotonic()
    t_last_print = t0
    trajectory_pts: list[list[float]] = []
    last_pose_count = 0

    try:
        while True:
            now = time.monotonic()
            if args.duration is not None and (now - t0) >= args.duration:
                break

            # Periodic status print
            if args.print_every > 0 and (now - t_last_print) >= args.print_every:
                stats = teleop.stats()
                pose = teleop.latest_pose
                if pose is not None:
                    pos = pose.translation
                    pstr = f"pos=({pos[0]:+6.3f}, {pos[1]:+6.3f}, {pos[2]:+6.3f})"
                else:
                    pstr = "pos=(no pose yet)"
                print(f"  t={now - t0:5.1f}s  n={stats['n_poses']:5d}  "
                      f"rate={stats.get('mean_hz', 0):5.1f} Hz  {pstr}")
                t_last_print = now

            # Push to rerun (only when a new pose arrived)
            if rr is not None:
                stats = teleop.stats()
                pose = teleop.latest_pose
                delta = teleop.latest_delta
                if pose is not None and stats["n_poses"] != last_pose_count:
                    last_pose_count = stats["n_poses"]
                    rr.set_time("time", duration=pose.t_host)
                    trajectory_pts.append(pose.translation.tolist())
                    rr.log("world/trajectory",
                           rr.LineStrips3D([trajectory_pts], colors=[0, 200, 255]))
                    rr.log("world/camera", rr.Transform3D(
                        translation=pose.translation.tolist(),
                        quaternion=pose.quaternion.tolist()))
                    _AXIS = 0.05
                    rr.log("world/camera/axes", rr.Arrows3D(
                        origins=[[0, 0, 0]] * 3,
                        vectors=[[_AXIS, 0, 0], [0, _AXIS, 0], [0, 0, _AXIS]],
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    ))
                    if delta is not None:
                        d_t_mm = float(np.linalg.norm(
                            [delta.dx, delta.dy, delta.dz])) * 1000.0
                        rr.log("delta/translation_mm", rr.Scalars(d_t_mm))

            time.sleep(0.005)
    except KeyboardInterrupt:
        print()

    print("Shutting down...")
    teleop.shutdown()
    final = teleop.stats()
    print(f"Final: n_poses={final['n_poses']}  mean_hz={final.get('mean_hz', 0)}  "
          f"p95={final.get('p95_inter_pose_ms', 0)} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
