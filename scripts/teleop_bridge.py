"""Bridge: grabette WS deltas + angle sensors → OpenArm/Gripette gRPC sim.

Subscribes to the grabette teleop stream and forwards each frame as a
gRPC command pair:
  - Camera-local Cartesian delta → ArmService.SendCartesianDelta
  - Angle sensor goal positions  → GripperService.SendMotorCommand

Conventions match the sim server's CartesianDelta interpretation
(see openarm_gripette_simu/arm_servicer.py): the WS delta is the
LeRobot §10.3 camera-local frame-to-frame delta, the integrator on the
server side composes it. dr6d is the 6D rotation encoding from
openarm_gripette_simu/rotation.py — first two rows of R, flattened.

Drift mitigation (the SLAM has non-zero per-step bias even when the
grabette is held still, which the server-side integrator will dutifully
accumulate into a slow arm drift):
  - IMU motion gate: a single state-poll task watches gyro/accel
    magnitudes and declares "static" when both are below threshold for
    a short window. While static, deltas are dropped entirely.
  - Per-step deadband: a small magnitude threshold zeroes out the
    position and/or rotation component below the SLAM noise floor.

Run on the same workstation that hosts the sim gRPC servers; expects
the openarm_gripette_simu repo as a sibling of grabette/.

Usage:
    uv run python scripts/teleop_bridge.py \
        --ws ws://rgrabette2:8000/api/teleop/stream \
        --api http://rgrabette2:8000 \
        --arm localhost:50052 --gripper localhost:50051
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Resolve sibling openarm_gripette_simu repo so we can import its proto
# stubs and the rotation conversion (same source of truth as the server).
# We add the INNER package directory (not the repo root) so `proto` and
# `rotation` are loaded as top-level — that skips the parent package's
# __init__.py which would otherwise pull in mujoco/placo etc. The bridge
# has no business depending on the sim's physics stack.
_THIS = Path(__file__).resolve()
_SIM_REPO = _THIS.parent.parent.parent / "openarm_gripette_simu"
_SIM_PKG = _SIM_REPO / "openarm_gripette_simu"
if not _SIM_PKG.exists():
    sys.exit(
        f"openarm_gripette_simu not found at {_SIM_REPO}. "
        "Bridge expects the sim repo as a sibling of grabette/."
    )
sys.path.insert(0, str(_SIM_PKG))

import grpc  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402
import websockets  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from proto import arm_pb2, arm_pb2_grpc  # noqa: E402
from proto import gripper_pb2, gripper_pb2_grpc  # noqa: E402
from rotation import rotation_matrix_to_6d  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
# Silence the per-request access logs from httpx / httpcore. The state_loop
# polls /api/state ~30 times/sec; without this, that one logger drowns out
# everything else (including our --debug output).
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logger = logging.getLogger("teleop_bridge")

# Standard gravity (m/s²). BMI088 reports raw accelerometer including g,
# so "static" means |accel| ≈ G regardless of orientation.
G_STANDARD = 9.80665


@dataclass
class SharedState:
    """Cross-task state. Single writer (state_loop) → single reader (arm_loop)."""
    is_static: bool = False
    last_motion_t: float = 0.0  # time.monotonic()


def quat_delta_to_dr6d(qx: float, qy: float, qz: float, qw: float) -> list[float]:
    """Convert a delta quaternion (xyzw) to the sim's 6D rotation format."""
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return rotation_matrix_to_6d(R).astype(np.float32).tolist()


def quat_angle_rad(qx: float, qy: float, qz: float, qw: float) -> float:
    """Magnitude of the rotation encoded by a (possibly non-unit) quaternion.

    Uses atan2 form for numerical stability near small angles.
    """
    vmag = math.sqrt(qx * qx + qy * qy + qz * qz)
    return 2.0 * math.atan2(vmag, abs(qw))


# Identity 6D rotation (first two rows of I3, flattened). Used to substitute
# the rotation component when the rotation deadband zeroes it out.
_IDENTITY_DR6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


async def arm_loop(
    ws_uri: str,
    arm_stub: arm_pb2_grpc.ArmServiceStub,
    shared: SharedState,
    cfg: argparse.Namespace,
    stats: dict,
) -> None:
    """Consume the WS stream and forward arm deltas via gRPC.

    The grabette WS sends at a fixed 30 Hz but SLAM produces poses at
    ~22 Hz, so ~30 % of WS messages re-send the previous delta unchanged
    (same `t` timestamp). Applying those duplicates to the server-side
    integrator amplifies real motion AND static drift by ~36 %, which
    presents as "the arm overshoots / drifts faster". We dedupe by `t`
    here — one delta per SLAM pose.
    """
    backoff = 1.0
    last_t = None
    while True:
        try:
            logger.info("connecting WS: %s", ws_uri)
            async with websockets.connect(ws_uri, ping_interval=10) as ws:
                logger.info("WS connected")
                backoff = 1.0
                last_t = None  # reset on reconnect
                async for raw in ws:
                    stats["ws_recv"] += 1
                    msg = json.loads(raw)

                    t = msg.get("t")
                    if t is not None and t == last_t and not msg.get("lost"):
                        stats["arm_skipped_duplicate"] += 1
                        continue
                    last_t = t

                    if msg.get("lost"):
                        stats["arm_skipped_lost"] += 1
                        if cfg.debug:
                            logger.debug("arm: LOST")
                        continue
                    if not msg.get("send"):
                        stats["arm_skipped_no_send"] += 1
                        continue
                    if not cfg.no_imu_gate and shared.is_static:
                        # Grabette is being held still — drop deltas so SLAM
                        # drift doesn't slowly walk the integrator.
                        stats["arm_skipped_static"] += 1
                        if cfg.debug:
                            logger.debug("arm: STATIC (skipped)")
                        continue

                    dx = float(msg["dx"])
                    dy = float(msg["dy"])
                    dz = float(msg["dz"])
                    qx = float(msg["dqx"])
                    qy = float(msg["dqy"])
                    qz = float(msg["dqz"])
                    qw = float(msg["dqw"])

                    # Per-step deadband — zero components below the SLAM
                    # noise floor. Position and rotation are independent.
                    pos_zeroed = False
                    rot_zeroed = False
                    pos_mag = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if pos_mag < cfg.pos_deadband_m:
                        dx = dy = dz = 0.0
                        pos_zeroed = True
                        stats["arm_pos_deadband"] += 1
                    rot_angle = quat_angle_rad(qx, qy, qz, qw)
                    if rot_angle < cfg.rot_deadband_rad:
                        rot_zeroed = True
                        stats["arm_rot_deadband"] += 1

                    if cfg.debug:
                        logger.debug(
                            "arm: pos_mag=%.4fmm rot=%.3fmrad   "
                            "pos_zeroed=%s rot_zeroed=%s",
                            pos_mag * 1000, rot_angle * 1000,
                            pos_zeroed, rot_zeroed,
                        )

                    if pos_zeroed and rot_zeroed:
                        # Nothing left to send — saves an RPC.
                        stats["arm_skipped_deadband"] += 1
                        continue

                    # Safety cap (after deadband, before send). SLAM
                    # re-acquisition spikes can produce >1m jumps; scale
                    # to keep direction.
                    if pos_mag > cfg.max_delta_m:
                        s = cfg.max_delta_m / pos_mag
                        dx, dy, dz = dx * s, dy * s, dz * s
                        stats["arm_clamped"] += 1

                    dr6d = (_IDENTITY_DR6D if rot_zeroed
                            else quat_delta_to_dr6d(qx, qy, qz, qw))

                    req = arm_pb2.CartesianDelta(
                        dx=dx, dy=dy, dz=dz, dr6d=dr6d,
                    )
                    # Sync gRPC call; ~1-3 ms locally, fine to block the loop.
                    try:
                        resp = arm_stub.SendCartesianDelta(req, timeout=0.5)
                        if not resp.success:
                            stats["arm_grpc_fail"] += 1
                            logger.warning("arm gRPC: %s", resp.error)
                        else:
                            stats["arm_sent"] += 1
                    except grpc.RpcError as e:
                        stats["arm_grpc_fail"] += 1
                        logger.warning("arm gRPC error: %s", e.code())
        except (OSError, websockets.WebSocketException) as e:
            logger.warning("WS connection lost (%s); reconnecting in %.1fs", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)


async def state_loop(
    api_base: str,
    grip_stub: gripper_pb2_grpc.GripperServiceStub | None,
    shared: SharedState,
    cfg: argparse.Namespace,
    stats: dict,
) -> None:
    """Single HTTP poller that handles both:
      - IMU motion gate (updates shared.is_static, read by arm_loop)
      - Gripper command (sends motor goals from angle sensors)
    Single endpoint hit per tick keeps load on the daemon minimal.
    """
    dt = 1.0 / cfg.state_rate_hz
    proximal_sign = -1.0 if cfg.invert_proximal else 1.0
    distal_sign = -1.0 if cfg.invert_distal else 1.0
    # Seed last_motion_t to "now" so we don't declare static immediately
    # before we've seen any IMU samples.
    shared.last_motion_t = time.monotonic()
    async with httpx.AsyncClient(base_url=api_base, timeout=2.0) as client:
        while True:
            t0 = time.monotonic()
            try:
                r = await client.get("/api/state")
                data = r.json()

                # ── IMU motion gate ────────────────────────────────────
                if not cfg.no_imu_gate:
                    imu = data.get("imu")
                    if imu is not None:
                        g = imu["gyro"]
                        a = imu["accel"]
                        gyro_mag = math.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)
                        accel_mag = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
                        accel_dev = abs(accel_mag - G_STANDARD)
                        if (gyro_mag > cfg.imu_gyro_thresh
                                or accel_dev > cfg.imu_accel_thresh):
                            shared.last_motion_t = t0
                        shared.is_static = (
                            (t0 - shared.last_motion_t) > cfg.static_window_s
                        )
                        if cfg.debug:
                            logger.debug(
                                "imu: |gyro|=%.4f rad/s  |accel|-g=%.3f m/s²  "
                                "static=%s  since_motion=%.2fs",
                                gyro_mag, accel_dev, shared.is_static,
                                t0 - shared.last_motion_t,
                            )

                # ── Gripper command ────────────────────────────────────
                if grip_stub is not None:
                    angle = data.get("angle")
                    if angle is not None:
                        m1 = proximal_sign * float(angle["proximal"])
                        m2 = distal_sign * float(angle["distal"])
                        req = gripper_pb2.MotorCommand(
                            motor1_goal=m1, motor2_goal=m2,
                        )
                        try:
                            resp = grip_stub.SendMotorCommand(req, timeout=0.5)
                            if not resp.success:
                                stats["grip_grpc_fail"] += 1
                                logger.warning("gripper gRPC: %s", resp.error)
                            else:
                                stats["grip_sent"] += 1
                        except grpc.RpcError as e:
                            stats["grip_grpc_fail"] += 1
                            logger.warning("gripper gRPC error: %s", e.code())
            except Exception as e:
                stats["state_http_fail"] += 1
                logger.debug("state poll error: %s", e)
            sleep = dt - (time.monotonic() - t0)
            if sleep > 0:
                await asyncio.sleep(sleep)


async def stats_printer(
    shared: SharedState,
    stats: dict,
    interval_s: float = 5.0,
) -> None:
    last = {k: 0 for k in stats}
    while True:
        await asyncio.sleep(interval_s)
        deltas = "  ".join(
            f"{k}=+{stats[k] - last[k]}" for k in sorted(stats)
            if stats[k] != last[k]
        )
        logger.info(
            "state=%s  %s",
            "STATIC" if shared.is_static else "moving",
            deltas if deltas else "(no activity)",
        )
        last = dict(stats)


async def main_async(args: argparse.Namespace) -> None:
    # Connect gRPC channels and ping. Bail loudly if either side is down —
    # better to fail at startup than to silently send into nothing.
    arm_channel = grpc.insecure_channel(args.arm)
    arm_stub = arm_pb2_grpc.ArmServiceStub(arm_channel)
    arm_stub.Ping(arm_pb2.ArmPingRequest(), timeout=2.0)
    logger.info("arm gRPC connected: %s", args.arm)

    grip_stub = None
    if not args.no_gripper:
        grip_channel = grpc.insecure_channel(args.gripper)
        grip_stub = gripper_pb2_grpc.GripperServiceStub(grip_channel)
        grip_stub.Ping(gripper_pb2.PingRequest(), timeout=2.0)
        logger.info("gripper gRPC connected: %s", args.gripper)

    shared = SharedState()
    stats = {
        "ws_recv": 0,
        "arm_sent": 0,
        "arm_skipped_duplicate": 0,
        "arm_skipped_no_send": 0,
        "arm_skipped_lost": 0,
        "arm_skipped_static": 0,
        "arm_skipped_deadband": 0,
        "arm_pos_deadband": 0,
        "arm_rot_deadband": 0,
        "arm_clamped": 0,
        "arm_grpc_fail": 0,
        "grip_sent": 0,
        "grip_grpc_fail": 0,
        "state_http_fail": 0,
    }

    tasks = [
        asyncio.create_task(arm_loop(args.ws, arm_stub, shared, args, stats)),
        asyncio.create_task(stats_printer(shared, stats)),
    ]
    # state_loop is needed whenever we have any consumer of /api/state —
    # either the gripper bridge or the IMU motion gate.
    if grip_stub is not None or not args.no_imu_gate:
        tasks.append(asyncio.create_task(
            state_loop(args.api, grip_stub, shared, args, stats)
        ))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ws", default="ws://rgrabette2:8000/api/teleop/stream",
        help="grabette teleop WS URL",
    )
    p.add_argument(
        "--api", default="http://rgrabette2:8000",
        help="grabette HTTP API base (polled for IMU + angle sensors)",
    )
    p.add_argument("--arm", default="localhost:50052", help="arm gRPC address")
    p.add_argument("--gripper", default="localhost:50051", help="gripper gRPC address")
    p.add_argument(
        "--max-delta-m", type=float, default=0.05,
        help="cap each per-step Cartesian delta magnitude (meters)",
    )
    p.add_argument(
        "--state-rate-hz", type=float, default=30.0,
        help="rate to poll /api/state (drives both IMU gate + gripper update)",
    )
    p.add_argument(
        "--no-gripper", action="store_true",
        help="skip the gripper bridge (arm only)",
    )
    p.add_argument(
        "--invert-proximal", action="store_true",
        help="negate the proximal angle before forwarding to motor1_goal",
    )
    p.add_argument(
        "--invert-distal", action="store_true",
        help="negate the distal angle before forwarding to motor2_goal",
    )
    # ── Drift mitigation ──────────────────────────────────────────────
    p.add_argument(
        "--no-imu-gate", action="store_true",
        help="disable the IMU-based motion gate (deltas always sent)",
    )
    p.add_argument(
        "--imu-gyro-thresh", type=float, default=0.05,
        help="gyro magnitude (rad/s) below which IMU is considered still",
    )
    p.add_argument(
        "--imu-accel-thresh", type=float, default=0.6,
        help="|accel-g| (m/s²) below which IMU is considered still. "
             "Default tuned for the BMI088 — its raw at-rest |accel| sits "
             "~0.4 m/s² off nominal g on this rig, so a tighter threshold "
             "would never trigger static.",
    )
    p.add_argument(
        "--static-window-s", type=float, default=0.15,
        help="how long IMU must read still before declaring static",
    )
    p.add_argument(
        "--pos-deadband-m", type=float, default=0.0005,
        help="zero per-step position delta below this magnitude (meters). "
             "Default 0.5 mm catches the static SLAM noise tail (saw spikes "
             "up to 0.4 mm at rest); deliberate hand motion is >1 mm/step.",
    )
    p.add_argument(
        "--rot-deadband-rad", type=float, default=0.005,
        help="zero per-step rotation below this angle magnitude (radians). "
             "Default 5 mrad (~0.3°) catches static SLAM rotation noise.",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="print per-tick IMU values, gate decisions, and per-delta decisions",
    )
    args = p.parse_args()

    if args.debug:
        logging.getLogger("teleop_bridge").setLevel(logging.DEBUG)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
