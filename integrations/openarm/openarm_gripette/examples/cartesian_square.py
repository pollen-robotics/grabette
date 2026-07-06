"""Move the EE in a square pattern using camera-LOCAL frame deltas.

This is the canonical end-to-end test of the camera-local delta convention
exposed by `arm_servicer.SendCartesianDelta` (sim) and
`grpc_server_real.SendCartesianDelta` (real). The server's integrator is:

    R_target_new   = R_target @ R_delta            (orientation)
    pos_target_new = pos_target + R_target @ Δpos  (position; LOCAL → world)

So `(dx, dy, dz)` is interpreted in the integrator's current camera frame,
NOT in world coordinates. To trace a clean shape with this script you read
the camera's local frame the same way the policy does, and send deltas
defined directly in that frame.

What the square traces (defaults):
  - Plane:     camera-local YZ plane  (image-down × optical-axis)
  - Orientation:    locked to start orientation (R_delta = identity every step)

Visual verification on the camera feed:
  - Edge 1 (camera moves -Z = backward along optical axis): scene zooms OUT.
  - Edge 2 (camera moves -Y = image-up): scene scrolls DOWN.
  - Edge 3 (camera moves +Z = forward along optical axis): scene zooms IN.
  - Edge 4 (camera moves +Y = image-down): scene scrolls UP.

If you instead see world-frame motion (e.g. always moves in the same horizontal
direction regardless of camera tilt), the integrator has reverted to world-
frame deltas — that's the regression the camera-local refactor was meant to
fix. See `feedback_action_deltas_camera_local` in memory.

Usage:
  uv run python examples/openarm_gripette/cartesian_square.py \\
      --arm_addr localhost:50052
  uv run python examples/openarm_gripette/cartesian_square.py \\
      --arm_addr <robot-ip>:50052 --gripper_addr <gripette-ip>:50051 --show_camera

  # Optional: trace the square in the camera-local XY plane (image-right ×
  # image-down) instead of the default YZ plane.
  uv run python examples/openarm_gripette/cartesian_square.py --plane xy
"""

import argparse
import logging
import threading
import time

import grpc
import numpy as np
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc

logger = logging.getLogger(__name__)

# Lazy import — only needed if --log_gripper_frame is set, and only when this
# script is run in an env that has placo + the URDF (i.e. the lerobot env).
_KIN = None


def _gripper_pos_from_joints(joints: np.ndarray) -> np.ndarray:
    """Run client-side FK on the URDF to get the gripper-frame world position
    for the given joint angles (rad). Cached singleton Kinematics."""
    global _KIN
    if _KIN is None:
        from openarm_gripette_simu import Kinematics
        from openarm_gripette_simu.kinematics import GRIPPER_FRAME
        _KIN = (Kinematics(), GRIPPER_FRAME)
    kin, gframe = _KIN
    T = kin.forward(joints, frame=gframe)
    return T[:3, 3]

# Identity rotation in 6D = first two columns of I_3.
# Sending this as `dr6d` makes the server's integrator apply
# `R_target_new = R_target @ I = R_target` — i.e. orientation stays locked
# at whatever the integrator was initialised to (FK at startup or after Reset).
IDENTITY_R6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

# Hard cap on per-step displacement for safety (meters).
MAX_PER_STEP_MM = 2.0

# Square geometry — small and slow for a safe first test.
SQUARE_HALF_SIZE = 0.03      # 3 cm half-edge → 6 cm square (safer default)
STEPS_PER_EDGE = 200         # 0.3 mm per step at default
COMMAND_HZ = 20              # 200 steps × 1/20 s = 10 s per edge

# `--tiny` preset: a 2 cm square. Used to rule out workspace-limit / wrist-
# singularity issues when the regular size hits one of them. If the tiny
# square is clean but the 6 cm one isn't, the failure is geometric (the larger
# trajectory leaves the IK-friendly region near the seed pose).
TINY_HALF_SIZE = 0.01        # 1 cm half-edge → 2 cm square

# Joint names in the order GetArmState returns them (`KIN_ARM_JOINT_NAMES`
# in the sim's kinematics).
JOINT_NAMES = [
    "r_arm_pitch",   # joint_1
    "r_arm_roll",    # joint_2
    "r_arm_yaw",     # joint_3
    "r_elbow",       # joint_4
    "r_wrist_yaw",   # joint_5
    "r_wrist_roll",  # joint_6
    "r_wrist_pitch", # joint_7
]

# Joint limits enforced by `OpenArm7FollowerConfig.RIGHT_DEFAULT_JOINTS_LIMITS`
# in degrees (clipped by the driver before the goal reaches the motor).
# These are TIGHTER than the URDF mechanical limits — Placo's IK doesn't see
# them, so a Cartesian target that requires a joint outside this range will
# silently miss on real hardware. Sim has no such layer, which is why a
# bigger square (e.g. half_size=0.05) can work in sim but miss on real.
DRIVER_JOINT_LIMITS_DEG = [
    (-75.0, 75.0),   # joint_1
    ( -9.0, 90.0),   # joint_2  (asymmetric)
    (-85.0, 85.0),   # joint_3
    (  0.0, 135.0),  # joint_4
    (-85.0, 85.0),   # joint_5
    (-40.0, 40.0),   # joint_6  (tight)
    (-80.0, 80.0),   # joint_7
]

# Flag a joint when its value is within this many degrees of either limit.
JOINT_SATURATION_FLAG_DEG = 5.0


def _saturation_flag(name: str, val_deg: float, lo: float, hi: float) -> str:
    """Return an empty string if the joint is comfortably inside its driver
    limits, otherwise a flag like `(j6 HI: 40.0)`."""
    margin = JOINT_SATURATION_FLAG_DEG
    if val_deg <= lo + margin:
        return f"(!{name.replace('r_', '')}<{lo:+.0f})"
    if val_deg >= hi - margin:
        return f"(!{name.replace('r_', '')}>{hi:+.0f})"
    return ""


def get_ee_pose(arm_stub) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch FK pose (pos, r6d) and arm joints (rad) from the server.

    NB: this is the server's FK-from-measured-joints reading, not the
    integrator's internal `_target_pos/_target_r6d`. We only use it for
    logging / sanity checks. The integrator is what actually drives motion.
    """
    state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
    return (np.array([state.x, state.y, state.z], dtype=np.float64),
            np.array(state.r6d, dtype=np.float64),
            np.array(state.joint_positions, dtype=np.float64))


def build_square_local_deltas(half: float, steps_per_edge: int, plane: str) -> list[tuple[float, float, float]]:
    """Per-step (dx, dy, dz) in the camera-LOCAL frame, tracing a closed square.

    Plane:
      yz  → camera-local YZ plane (image-down × optical-axis): forward/back × up/down
      xy  → camera-local XY plane (image-right × image-down): horizontal in the image
    """
    step = 2 * half / steps_per_edge
    if plane == "yz":
        edges = [
            (0.0, 0.0, -step),  # along -Z (optical axis, backward) → scene zooms OUT
            (0.0, -step, 0.0),  # along -Y (image-up) → scene scrolls DOWN
            (0.0, 0.0, +step),  # along +Z (forward) → scene zooms IN
            (0.0, +step, 0.0),  # along +Y (image-down) → scene scrolls UP
        ]
    elif plane == "xy":
        edges = [
            (+step, 0.0, 0.0),  # along +X (image-right) → scene scrolls LEFT
            (0.0, +step, 0.0),  # along +Y (image-down) → scene scrolls UP
            (-step, 0.0, 0.0),  # along -X (image-left) → scene scrolls RIGHT
            (0.0, -step, 0.0),  # along -Y (image-up) → scene scrolls DOWN
        ]
    else:
        raise ValueError(f"Unknown plane: {plane!r} (expected 'yz' or 'xy')")

    deltas = []
    for edge in edges:
        deltas.extend([edge] * steps_per_edge)
    return deltas


def camera_display_thread(gripper_addr: str, stop_event: threading.Event):
    import cv2
    from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc

    channel = grpc.insecure_channel(gripper_addr)
    stub = gripper_pb2_grpc.GripperServiceStub(channel)
    try:
        for frame in stub.StreamState(gripper_pb2.StreamRequest()):
            if stop_event.is_set():
                break
            img = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow("Gripette camera", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
    except grpc.RpcError as e:
        if not stop_event.is_set():
            logger.warning(f"Camera stream ended: {e}")
    finally:
        channel.close()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Trace a camera-LOCAL Cartesian square via gRPC")
    p.add_argument("--arm_addr", type=str, default="localhost:50052")
    p.add_argument("--gripper_addr", type=str, default="localhost:50051")
    p.add_argument("--show_camera", action="store_true")
    p.add_argument("--loops", type=int, default=0, help="0 = infinite")
    p.add_argument("--half_size", type=float, default=SQUARE_HALF_SIZE,
                   help=f"Square half-edge in meters (default: {SQUARE_HALF_SIZE})")
    p.add_argument("--tiny", action="store_true",
                   help=f"Shortcut for --half_size {TINY_HALF_SIZE} (2 cm square). "
                   "Use when the default size leaves the IK-friendly region "
                   "around the seed joint config.")
    p.add_argument("--steps_per_edge", type=int, default=STEPS_PER_EDGE)
    p.add_argument("--fps", type=float, default=COMMAND_HZ)
    p.add_argument("--plane", type=str, default="yz", choices=["yz", "xy"],
                   help="Camera-local plane in which to trace the square (default: yz)")
    p.add_argument("--log_gripper_frame", action="store_true",
                   help="Also log the gripper-tip position (FK at the 'gripper' "
                        "URDF frame). The IK locks the camera-site (which the "
                        "policy uses), but you watch the gripper tip — these "
                        "two diverge when wrist joints swing in the null space.")
    args = p.parse_args()
    if args.tiny:
        args.half_size = TINY_HALF_SIZE
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    # Safety check on per-step displacement.
    step_mm = 2 * args.half_size / args.steps_per_edge * 1000
    if step_mm > MAX_PER_STEP_MM:
        raise ValueError(
            f"Per-step displacement {step_mm:.2f} mm exceeds safety limit "
            f"{MAX_PER_STEP_MM} mm. Increase --steps_per_edge or reduce --half_size."
        )
    logger.info(
        f"Camera-local square in '{args.plane}' plane: {step_mm:.2f} mm/step @ "
        f"{args.fps:.0f} Hz → {step_mm * args.fps:.1f} mm/s, "
        f"{args.steps_per_edge * 4 / args.fps:.1f} s per loop"
    )

    channel = grpc.insecure_channel(args.arm_addr)
    arm_stub = arm_pb2_grpc.ArmServiceStub(channel)
    ping = arm_stub.Ping(arm_pb2.ArmPingRequest())
    logger.info(f"Server: {ping.status} (uptime: {ping.uptime_seconds:.1f}s)")

    stop_event = threading.Event()
    cam_thread = None
    if args.show_camera:
        cam_thread = threading.Thread(
            target=camera_display_thread,
            args=(args.gripper_addr, stop_event),
            daemon=True,
        )
        cam_thread.start()

    # Log the starting pose. We do NOT rotate deltas through it — the server
    # already applies its own integrator rotation `R_target @ Δpos`. Reading
    # the FK pose here is just for the human operator's situational awareness.
    start_pos, start_r6d, start_joints = get_ee_pose(arm_stub)
    logger.info(f"Start EE position (world): "
                f"[{start_pos[0]:+.3f}, {start_pos[1]:+.3f}, {start_pos[2]:+.3f}] m")
    start_joints_deg = np.rad2deg(start_joints)
    logger.info(f"Start joints (deg): "
                f"j1={start_joints_deg[0]:+.1f} j2={start_joints_deg[1]:+.1f} "
                f"j3={start_joints_deg[2]:+.1f} j4={start_joints_deg[3]:+.1f} "
                f"j5={start_joints_deg[4]:+.1f} j6={start_joints_deg[5]:+.1f} "
                f"j7={start_joints_deg[6]:+.1f}")
    logger.info("Driver joint limits (deg, applied by OpenArm7Follower.send_action):")
    for name, (lo, hi) in zip(JOINT_NAMES, DRIVER_JOINT_LIMITS_DEG):
        logger.info(f"  {name:14s} ∈ [{lo:+6.1f}, {hi:+6.1f}]")
    logger.info("Orientation will be locked (sending identity R_delta every step).")

    # Mirror the server's integrator on the client so we can verify that the
    # measured EE position (from GetArmState = FK of measured joints) actually
    # tracks the commanded cumulative target. Since dr6d = identity every step,
    # the server's R_target stays at its startup value, which equals the FK
    # rotation we just read into start_r6d. We can rebuild the same matrix on
    # the client side and accumulate position locally.
    row0 = np.array(start_r6d[:3])
    row1 = np.array(start_r6d[3:6])
    row2 = np.cross(row0, row1)
    R_target = np.stack([row0, row1, row2], axis=0)
    expected_pos = start_pos.copy().astype(np.float64)
    # Total cumulative position error stats (max + final-of-each-edge).
    max_pos_err_mm = 0.0
    # Starting image-right direction (camera +X in world), captured at step 0,
    # used to compute roll angle around the optical axis on subsequent samples.
    _start_image_right: list = []
    max_roll_deg = 0.0

    deltas = build_square_local_deltas(args.half_size, args.steps_per_edge, args.plane)
    total_steps = len(deltas)
    dt = 1.0 / args.fps

    try:
        loop_idx = 0
        while not stop_event.is_set():
            for i, (dx, dy, dz) in enumerate(deltas):
                if stop_event.is_set():
                    break
                t0 = time.perf_counter()

                arm_stub.SendCartesianDelta(arm_pb2.CartesianDelta(
                    dx=dx, dy=dy, dz=dz, dr6d=IDENTITY_R6D,
                ))
                # Track expected cumulative target client-side. Identical math
                # to the server's integrator (`_target_pos += R_target @ Δpos`).
                expected_pos = expected_pos + R_target @ np.array([dx, dy, dz])

                if i % (args.steps_per_edge // 4) == 0:
                    pos, r6d, joints = get_ee_pose(arm_stub)
                    pos_err_mm = float(np.linalg.norm(pos - expected_pos) * 1000)
                    max_pos_err_mm = max(max_pos_err_mm, pos_err_mm)
                    # `r6d` is the first two ROWS of the 3x3 rotation matrix
                    # (lerobot's `rotation_matrix_to_6d` slices [..., :2, :]).
                    # Rebuild rows, recover the third row via cross product,
                    # then take the THIRD COLUMN — that's the camera optical
                    # axis (+Z in camera-local) expressed in world coords.
                    row0 = np.array(r6d[:3])
                    row1 = np.array(r6d[3:6])
                    row2 = np.cross(row0, row1)
                    R = np.stack([row0, row1, row2], axis=0)
                    image_right = R[:, 0]   # camera +X in world
                    image_down  = R[:, 1]   # camera +Y in world
                    optical     = R[:, 2]   # camera +Z (optical axis) in world
                    # Roll angle around the optical axis: change of image-right
                    # direction from its starting orientation, projected into
                    # the plane perpendicular to the (fixed) optical axis.
                    # Captures camera "roll" that leaves the optical axis
                    # invariant but rotates the image — what a wrist joint
                    # aligned with the optical axis produces.
                    if i == 0 and loop_idx == 0:
                        # Cache the starting image-right; subsequent samples
                        # measure the roll relative to it.
                        _start_image_right.append(image_right.copy())
                    start_ir = _start_image_right[0]
                    # signed roll angle (degrees) about the current optical axis
                    proj = image_right - np.dot(image_right, optical) * optical
                    proj_start = start_ir - np.dot(start_ir, optical) * optical
                    proj /= max(np.linalg.norm(proj), 1e-9)
                    proj_start /= max(np.linalg.norm(proj_start), 1e-9)
                    cos_a = float(np.clip(np.dot(proj, proj_start), -1, 1))
                    sign = float(np.sign(np.dot(np.cross(proj_start, proj), optical)))
                    roll_deg = sign * np.degrees(np.arccos(cos_a))
                    deg = np.rad2deg(joints)
                    # Flag any joint within `JOINT_SATURATION_FLAG_DEG` of a
                    # driver limit. Joint limits below are those enforced by
                    # `OpenArm7FollowerConfig.RIGHT_DEFAULT_JOINTS_LIMITS`.
                    flags = [
                        _saturation_flag(name, val, lo, hi)
                        for (name, val, (lo, hi)) in zip(
                            JOINT_NAMES, deg, DRIVER_JOINT_LIMITS_DEG
                        )
                    ]
                    flag_str = " ".join(flags)
                    gripper_str = ""
                    if args.log_gripper_frame:
                        try:
                            g = _gripper_pos_from_joints(joints)
                            gripper_str = f" gripper [{g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f}]"
                        except Exception as e:
                            gripper_str = f" gripper=<err: {e}>"
                    max_roll_deg = max(max_roll_deg, abs(roll_deg))
                    logger.info(
                        f"  loop {loop_idx} step {i:>4d}/{total_steps}: "
                        f"EE [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] "
                        f"expect [{expected_pos[0]:+.3f}, {expected_pos[1]:+.3f}, {expected_pos[2]:+.3f}] "
                        f"err={pos_err_mm:5.1f}mm{gripper_str}  "
                        f"optical [{optical[0]:+.2f}, {optical[1]:+.2f}, {optical[2]:+.2f}] "
                        f"img_right [{image_right[0]:+.2f}, {image_right[1]:+.2f}, {image_right[2]:+.2f}] "
                        f"roll={roll_deg:+6.1f}°  "
                        f"joints (deg): "
                        f"j1={deg[0]:+6.1f} j2={deg[1]:+6.1f} j3={deg[2]:+6.1f} "
                        f"j4={deg[3]:+6.1f} j5={deg[4]:+6.1f} j6={deg[5]:+6.1f} "
                        f"j7={deg[6]:+6.1f}{(' ' + flag_str) if flag_str.strip() else ''}"
                    )

                elapsed = time.perf_counter() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            loop_idx += 1
            if args.loops > 0 and loop_idx >= args.loops:
                logger.info(f"Completed {args.loops} loop(s)")
                final_err_mm = float(np.linalg.norm(
                    get_ee_pose(arm_stub)[0] - expected_pos
                ) * 1000)
                logger.info(
                    f"Position-error summary: peak {max_pos_err_mm:.1f} mm, "
                    f"final {final_err_mm:.1f} mm. "
                    f"(<2 mm = closed loop is tracking; "
                    f">5 mm = IK can't reach target or motor not tracking)"
                )
                logger.info(
                    f"Camera-roll summary: peak |roll| = {max_roll_deg:.1f}° "
                    f"around the optical axis. "
                    f"(<2° = orientation truly held; "
                    f">5° = IK is using camera roll as the null-space DOF — "
                    f"a posture task on the wrist joints would fix this.)"
                )
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        stop_event.set()
        if cam_thread is not None:
            cam_thread.join(timeout=2.0)
        channel.close()


if __name__ == "__main__":
    main()
