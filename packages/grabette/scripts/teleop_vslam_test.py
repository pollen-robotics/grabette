#!/usr/bin/env python3
"""Test depthai's live RTABMapVIO node on the OAK-D SR.

Goal: validate that the live VIO path (Path B in docs/teleop_design.md) is
viable for teleop — measures framerate, latency, and produces a TUM-format
trajectory that can be compared against our offline rtabmap pipeline on the
same recorded data.

Pipeline:
    CAM_B ─┐
           ├─ StereoDepth (ROBOTICS preset, LR check, rectify) ─┬─ rect ─┐
    CAM_C ─┘                                                    └ depth ─┤
                                                                         ├─ RTABMapVIO ─→ transform
    IMU (accel + gyro RAW @ 200Hz) ─────────────────────────────── imu ──┘

Outputs in `--output-dir`:
    live_trajectory.tum     TUM-format poses: t tx ty tz qx qy qz qw
    deltas.csv              Camera-local deltas (frame-to-frame, optical frame)
    summary.json            Framerate, latency, drift stats

With --save-frames, also writes (matching `oakd.py` recording layout, so
offline rtabmap can be re-run on the same data for apples-to-apples comparison):
    oakd_left.mp4           rectified left H.264 (1280×800 rectified to 640×400)
    oakd_depth/<seq>.png    uint16 depth in mm
    oakd_left_timestamps.json
    oakd_depth_timestamps.json
    oakd_imu.json
    oakd_calib_offline.json

Usage:
    uv run python scripts/teleop_vslam_test.py --duration 30
    uv run python scripts/teleop_vslam_test.py --duration 30 --save-frames

After running with --save-frames, convert + run offline rtabmap on the
same data:
    python ../grabette-data/scripts/rgbd_slam/convert_episode_to_oak.py -i <out>
    python ../grabette-data/scripts/rgbd_slam/run_oak_slam.py -i <out>

Then compare with evo (`pip install evo`):
    evo_ape tum <out>/live_trajectory.tum <out>/camera_trajectory_as_tum.tum --align
    evo_rpe tum <out>/live_trajectory.tum <out>/camera_trajectory_as_tum.tum --align
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from scipy.spatial.transform import Rotation

WIDTH, HEIGHT = 640, 400
FPS = 30
IMU_HZ = 200

# Basalt's transform output is in FRD camera-local (X-forward, Y-right, Z-down).
# Our project standard (and what offline rtabmap produces after R_camera_fix)
# is OpenCV optical RDF (X-right, Y-down, Z-forward). Right-multiply each pose
# by this matrix to re-express its local basis as optical:
#   pose_optical = pose_FRD * BASALT_TO_OPTICAL
# Columns of BASALT_TO_OPTICAL are the optical basis vectors expressed in FRD:
#   RDF +X (right)   = FRD +Y → (0, 1, 0)
#   RDF +Y (down)    = FRD +Z → (0, 0, 1)
#   RDF +Z (forward) = FRD +X → (1, 0, 0)
BASALT_TO_OPTICAL = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=30.0, help="recording duration (s)")
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("/tmp/teleop_vslam_test"),
                    help="where to write outputs")
    ap.add_argument("--save-frames", action="store_true",
                    help="also save rect+depth+imu for offline rtabmap comparison")
    ap.add_argument("--print-every", type=int, default=30,
                    help="print a status line every N poses (0 = silent)")
    ap.add_argument("--backend", choices=["rtabmap", "basalt"], default="rtabmap",
                    help="VIO backend node to test (depthai 3.6.1 exposes both)")
    ap.add_argument("--rerun-host", default=None,
                    help="Stream live trajectory + deltas to a rerun viewer at "
                         "this host:port (e.g. 192.168.1.5:9876). Start the viewer "
                         "with `rerun --port 9876 --bind 0.0.0.0` on the target host. "
                         "If omitted, no live viz.")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out}")

    # ── Build pipeline ─────────────────────────────────────────────────────────
    pipeline = dai.Pipeline()

    camB = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    camC = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    leftIn  = camB.requestOutput((WIDTH, HEIGHT), type=dai.ImgFrame.Type.GRAY8, fps=FPS)
    rightIn = camC.requestOutput((WIDTH, HEIGHT), type=dai.ImgFrame.Type.GRAY8, fps=FPS)

    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(
        [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
        IMU_HZ,
    )
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # StereoDepth is needed for rtabmap (it consumes rect+depth) and for
    # --save-frames in any backend (the offline rtabmap pipeline needs the
    # same rect+depth+imu layout). Basalt-only without --save-frames skips it
    # entirely, which is the lightest pipeline.
    need_stereo = (args.backend == "rtabmap") or args.save_frames
    if need_stereo:
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=leftIn, right=rightIn,
            presetMode=dai.node.StereoDepth.PresetMode.ROBOTICS,
        )
        stereo.setOutputSize(WIDTH, HEIGHT)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setRectifyEdgeFillColor(0)
        stereo.enableDistortionCorrection(True)
        stereo.initialConfig.setLeftRightCheckThreshold(10)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_B)
    else:
        stereo = None

    if args.backend == "rtabmap":
        vio = pipeline.create(dai.node.RTABMapVIO)
        stereo.rectifiedLeft.link(vio.rect)
        stereo.depth.link(vio.depth)
        imu.out.link(vio.imu)
    else:  # basalt
        vio = pipeline.create(dai.node.BasaltVIO)
        # Tell Basalt the IMU rate. Without this, the IMU integration is
        # timestep-blind, which produces drift that looks like "pure IMU"
        # because the optimizer can't reconcile visual and inertial data.
        vio.setImuUpdateRate(IMU_HZ)
        leftIn.link(vio.left)
        rightIn.link(vio.right)
        imu.out.link(vio.imu)

    q_pose = vio.transform.createOutputQueue(maxSize=8, blocking=False)
    print(f"VIO backend: {args.backend}  (StereoDepth branch: {'yes' if need_stereo else 'no'})")

    # Optional: save raw frames + IMU for offline rerun. With --backend basalt,
    # the StereoDepth branch is parallel to BasaltVIO — VIO consumes raw stereo,
    # StereoDepth produces the rect+depth we save for offline comparison.
    if args.save_frames:
        # H.264 encode rectified left for video
        enc = pipeline.create(dai.node.VideoEncoder).build(
            input=stereo.rectifiedLeft,
            bitrate=8_000_000,
            frameRate=float(FPS),
            profile=dai.VideoEncoderProperties.Profile.H264_MAIN,
            keyframeFrequency=FPS,
        )
        enc.setNumBFrames(0)
        q_h264 = enc.out.createOutputQueue(maxSize=32, blocking=False)
        q_depth = stereo.depth.createOutputQueue(maxSize=8, blocking=False)
        q_imu = imu.out.createOutputQueue(maxSize=200, blocking=False)
        (out / "oakd_depth").mkdir(exist_ok=True)
        h264_path = out / "oakd_left.h264"
        h264_fp = open(h264_path, "wb")
        left_ts_samples: list[dict] = []
        depth_ts_samples: list[dict] = []
        imu_samples: list[dict] = []
    else:
        q_h264 = q_depth = q_imu = None
        h264_path = None
        h264_fp = None
        left_ts_samples = depth_ts_samples = imu_samples = []  # unused

    # ── Optional: live visualization via rerun ─────────────────────────────────
    rr = None
    if args.rerun_host:
        try:
            import rerun as rr_mod
            rr_mod.init(f"grabette_teleop_{args.backend}", spawn=False)
            # rerun renamed its connect APIs across versions. Try the new gRPC
            # form first (rerun >=0.19), then the older tcp form, then the
            # legacy connect() — whichever the installed SDK exposes.
            if hasattr(rr_mod, "connect_grpc"):
                rr_mod.connect_grpc(f"rerun+http://{args.rerun_host}/proxy")
            elif hasattr(rr_mod, "connect_tcp"):
                rr_mod.connect_tcp(args.rerun_host)
            elif hasattr(rr_mod, "connect"):
                rr_mod.connect(args.rerun_host)
            else:
                raise RuntimeError(
                    f"no known connect API on rerun {rr_mod.__version__}"
                )
            rr_mod.log("world", rr_mod.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr_mod.log("world/axes", rr_mod.Arrows3D(
                origins=[[0, 0, 0]] * 3,
                vectors=[[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ), static=True)
            rr = rr_mod
            print(f"rerun: streaming to {args.rerun_host}")
        except ImportError:
            print("rerun-sdk not installed — install with: uv pip install rerun-sdk")
        except Exception as e:
            print(f"rerun: failed to connect to {args.rerun_host}: {e}")

    trajectory_pts: list[list[float]] = []  # accumulated positions for the line strip

    # ── Start pipeline ─────────────────────────────────────────────────────────
    print("Starting pipeline...")
    pipeline.start()

    # Dump calib for offline comparison (same flat schema as oakd.py)
    if args.save_frames:
        device = pipeline.getDefaultDevice()
        calib = device.readCalibration()
        intr = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, WIDTH, HEIGHT)
        imu_extr = calib.getImuToCameraExtrinsics(dai.CameraBoardSocket.CAM_B, True)
        baseline = calib.getBaselineDistance(
            dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_B) / 100.0
        (out / "oakd_calib_offline.json").write_text(json.dumps({
            "width": WIDTH, "height": HEIGHT,
            "fx": intr[0][0], "fy": intr[1][1],
            "cx": intr[0][2], "cy": intr[1][2],
            "baseline": baseline, "imu_to_cam": imu_extr,
        }, indent=2))

    # ── Run loop ───────────────────────────────────────────────────────────────
    tum_path = out / "live_trajectory.tum"
    deltas_path = out / "deltas.csv"
    tum_fp = tum_path.open("w")
    deltas_fp = deltas_path.open("w")
    deltas_fp.write("idx,t_host,dx,dy,dz,dqx,dqy,dqz,dqw\n")

    n_poses = 0
    n_frames_saved = 0
    pose_arrival_times: list[float] = []
    process_latencies: list[float] = []
    prev_pose: np.ndarray | None = None  # 4x4
    t_start = time.monotonic()
    t_end = t_start + args.duration

    while time.monotonic() < t_end:
        # Drain pose queue
        while True:
            try:
                p = q_pose.tryGet()
            except Exception:
                p = None
            if p is None:
                break
            t_arrive = time.monotonic()
            t_device = p.getTimestamp().total_seconds()
            # TUM timestamps in HOST monotonic seconds (matches the offline
            # rtabmap pipeline, which uses host_ms via oakd.py / SyncManager).
            # Using device time here causes evo to find zero matches because
            # the clocks have unrelated reference points.
            t_tum = t_arrive
            tx, ty, tz = p.getTranslation().x, p.getTranslation().y, p.getTranslation().z
            qx, qy, qz, qw = (p.getQuaternion().qx, p.getQuaternion().qy,
                              p.getQuaternion().qz, p.getQuaternion().qw)

            # Build the pose matrix, then fix the camera frame convention
            # so downstream (TUM, deltas, rerun) all see OpenCV optical RDF
            # regardless of which VIO backend produced the pose.
            T_curr = np.eye(4)
            T_curr[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T_curr[:3, 3] = [tx, ty, tz]
            if args.backend == "basalt":
                # Right-multiply: change camera-local basis FRD → optical.
                # Translation unchanged (world-frame position of camera origin).
                T_curr[:3, :3] = T_curr[:3, :3] @ BASALT_TO_OPTICAL
                qx, qy, qz, qw = Rotation.from_matrix(T_curr[:3, :3]).as_quat()

            tum_fp.write(f"{t_tum:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                         f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

            # Camera-local delta (per LeRobot §10.3 convention).
            # T_curr = world←cam_curr.  delta = T_prev⁻¹ · T_curr
            if prev_pose is None:
                d = np.eye(4)
            else:
                d = np.linalg.inv(prev_pose) @ T_curr
            dt = d[:3, 3]
            dq = Rotation.from_matrix(d[:3, :3]).as_quat()
            deltas_fp.write(f"{n_poses},{t_arrive - t_start:.6f},"
                            f"{dt[0]:.6f},{dt[1]:.6f},{dt[2]:.6f},"
                            f"{dq[0]:.6f},{dq[1]:.6f},{dq[2]:.6f},{dq[3]:.6f}\n")
            prev_pose = T_curr

            # Live viz: push to rerun if connected
            if rr is not None:
                t_rel = t_arrive - t_start
                rr.set_time("time", duration=t_rel)
                trajectory_pts.append([tx, ty, tz])
                rr.log("world/trajectory", rr.LineStrips3D([trajectory_pts], colors=[0, 200, 255]))
                # Position+orient the camera entity (Transform3D inherits to children).
                rr.log("world/camera", rr.Transform3D(
                    translation=[tx, ty, tz], quaternion=[qx, qy, qz, qw],
                ))
                # Draw the camera's local axes via Arrows3D — children inherit the
                # Transform3D above, so these appear at the camera's pose in world.
                # Done this way (not via Transform3D's axis_length kwarg) because
                # axis_length is only available on newer rerun versions.
                _AXIS = 0.05
                rr.log("world/camera/axes", rr.Arrows3D(
                    origins=[[0, 0, 0]] * 3,
                    vectors=[[_AXIS, 0, 0], [0, _AXIS, 0], [0, 0, _AXIS]],
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                ))
                # Delta scalars (frame-to-frame motion magnitude)
                d_t_mm = float(np.linalg.norm(dt)) * 1000.0
                d_r_deg = float(np.linalg.norm(
                    Rotation.from_quat(dq).as_rotvec()
                )) * (180.0 / np.pi)
                rr.log("delta/translation_mm", rr.Scalars(d_t_mm))
                rr.log("delta/rotation_deg", rr.Scalars(d_r_deg))

            pose_arrival_times.append(t_arrive)
            process_latencies.append(t_arrive - t_start - t_device + pose_arrival_times[0] - t_start)
            n_poses += 1
            if args.print_every and n_poses % args.print_every == 0:
                if len(pose_arrival_times) >= 2:
                    intervals = np.diff(pose_arrival_times[-30:])
                    hz = 1.0 / intervals.mean() if intervals.mean() > 0 else 0.0
                else:
                    hz = 0.0
                print(f"  [{n_poses:5d}] t={t_arrive - t_start:5.1f}s  rate={hz:5.1f} Hz  "
                      f"pos=({tx:+6.3f},{ty:+6.3f},{tz:+6.3f})")

        # Drain & save frames/imu if requested
        if args.save_frames:
            while True:
                pkt = q_h264.tryGet()
                if pkt is None:
                    break
                h264_fp.write(pkt.getData())
                left_ts_samples.append({"seq": int(pkt.getSequenceNum()),
                                        "host_ms": time.monotonic() * 1000.0})
            while True:
                d = q_depth.tryGet()
                if d is None:
                    break
                seq = int(d.getSequenceNum())
                cv2.imwrite(str(out / "oakd_depth" / f"{seq:08d}.png"),
                            d.getCvFrame(), [cv2.IMWRITE_PNG_COMPRESSION, 1])
                depth_ts_samples.append({"seq": seq, "host_ms": time.monotonic() * 1000.0})
                n_frames_saved += 1
            while True:
                m = q_imu.tryGet()
                if m is None:
                    break
                host_ms = time.monotonic() * 1000.0
                for pk in m.packets:
                    if hasattr(pk, "acceleroMeter") and pk.acceleroMeter:
                        a = pk.acceleroMeter
                        imu_samples.append({"kind": "accel", "host_ms": host_ms,
                                            "value": [a.x, a.y, a.z]})
                    if hasattr(pk, "gyroscope") and pk.gyroscope:
                        g = pk.gyroscope
                        imu_samples.append({"kind": "gyro", "host_ms": host_ms,
                                            "value": [g.x, g.y, g.z]})
        time.sleep(0.001)

    # ── Shutdown ───────────────────────────────────────────────────────────────
    print("Stopping pipeline...")
    pipeline.stop()
    tum_fp.close()
    deltas_fp.close()

    if args.save_frames:
        h264_fp.close()
        # Mux to MP4
        mp4_path = out / "oakd_left.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-fflags", "+genpts", "-r", f"{FPS:.3f}",
            "-i", str(h264_path), "-c", "copy", "-video_track_timescale", "90000",
            str(mp4_path),
        ], capture_output=True, text=True)
        h264_path.unlink(missing_ok=True)
        (out / "oakd_left_timestamps.json").write_text(json.dumps({"samples": left_ts_samples}))
        (out / "oakd_depth_timestamps.json").write_text(json.dumps({"samples": depth_ts_samples}))
        (out / "oakd_imu.json").write_text(json.dumps({"samples": imu_samples}))

    # ── Stats summary ──────────────────────────────────────────────────────────
    if pose_arrival_times:
        intervals = np.diff(pose_arrival_times)
        mean_hz = 1.0 / intervals.mean() if intervals.mean() > 0 else 0.0
        p95_interval_ms = float(np.percentile(intervals, 95)) * 1000.0
        p99_interval_ms = float(np.percentile(intervals, 99)) * 1000.0
    else:
        mean_hz = 0.0
        p95_interval_ms = p99_interval_ms = 0.0

    summary = {
        "duration_s": args.duration,
        "n_poses": n_poses,
        "mean_hz": round(mean_hz, 2),
        "p95_inter_pose_ms": round(p95_interval_ms, 2),
        "p99_inter_pose_ms": round(p99_interval_ms, 2),
        "frames_saved": n_frames_saved,
        "tum_path": str(tum_path),
        "deltas_path": str(deltas_path),
        "saved_for_offline_comparison": args.save_frames,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print()
    print("=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if mean_hz < 15.0:
        print("  ⚠ framerate below 15 Hz target — teleop will feel sluggish")
    else:
        print("  ✓ framerate above 15 Hz target")

    return 0


if __name__ == "__main__":
    sys.exit(main())
