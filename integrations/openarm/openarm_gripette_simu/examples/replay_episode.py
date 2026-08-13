"""GT-delta replay: stream a recorded episode's actions through the arm, NO model.

The decisive data↔deployment frame test. The converted dataset's actions are
camera-local deltas; the arm server integrates them in its CONTROL_FRAME. If
the recorded frame and the control frame agree, replaying an episode's actions
reproduces the demo's motion *shape* relative to the arm's start pose — a
coherent reach + close (and, with the object placed as in the demo relative to
the gripper, an actual grasp). If instead the arm veers off in a consistently
skewed direction, the control frame / axis convention does not match the data
— a deployment-transform bug, isolated from the policy entirely.

Protocol:
  1. Pick a KEPT episode from the converted (…_cartesian) dataset.
  2. Place the object roughly where that demo had it relative to the gripper
     start pose (watch the episode video to see).
  3. Run this script; watch the motion. Same shape as the demo video → frames
     agree (a policy miss is then precision/coverage, not transforms). Skewed
     direction → fix the control frame before blaming the policy.

Uses the SAME gRPC calls as evaluate.py (SendCartesianDelta / SendMotorCommand),
so it exercises the exact integration path the policy runs through.

Usage:
  uv run --extra eval python examples/replay_episode.py \\
      --dataset_repo_id <user>/<dataset>_cartesian --episode 0 \\
      --arm_addr localhost:50052 --gripper_addr localhost:50051
"""

import argparse
import logging
import time

import grpc
import numpy as np

from lerobot.datasets import LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Replay a recorded episode's delta actions on the arm (no model)")
    p.add_argument("--dataset_repo_id", required=True, help="CONVERTED (11D delta) dataset")
    p.add_argument("--dataset_root", default=None, help="Local dataset root (else HF cache/Hub)")
    p.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    p.add_argument("--arm_addr", default="localhost:50052", help="ArmService gRPC address")
    p.add_argument("--gripper_addr", default="localhost:50051", help="GripperService gRPC address")
    p.add_argument("--fps", type=float, default=50.0,
                   help="Playback rate. Dataset native is 50; use e.g. 25 for a slower, safer replay "
                        "(same trajectory, half speed).")
    p.add_argument("--max_steps", type=int, default=None, help="Stop after N steps (default: full episode)")
    p.add_argument("--no_reset", action="store_true", help="Skip arm_stub.Reset() before replaying")
    return p.parse_args()


def main():
    args = parse_args()

    ds = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, episodes=[args.episode])
    hf = ds.hf_dataset.select_columns(["action"])
    actions = np.stack([np.asarray(a, dtype=np.float32) for a in hf["action"]])
    if actions.shape[1] != 11:
        raise SystemExit(f"Expected 11D delta actions, got {actions.shape[1]}D — "
                         f"point this at the CONVERTED (…_cartesian) dataset.")
    n = len(actions) if args.max_steps is None else min(len(actions), args.max_steps)
    logger.info(f"Episode {args.episode}: {len(actions)} steps (replaying {n} at {args.fps:.0f} fps)")
    logger.info(f"  |Δpos| mean {np.linalg.norm(actions[:, :3], axis=1).mean() * 1000:.1f} mm  "
                f"max {np.linalg.norm(actions[:, :3], axis=1).max() * 1000:.1f} mm")

    from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc, gripper_pb2, gripper_pb2_grpc
    arm_stub = arm_pb2_grpc.ArmServiceStub(grpc.insecure_channel(args.arm_addr))
    gripper_stub = gripper_pb2_grpc.GripperServiceStub(grpc.insecure_channel(args.gripper_addr))

    if not args.no_reset:
        resp = arm_stub.Reset(arm_pb2.ResetRequest())
        if not resp.success:
            raise SystemExit(f"Reset failed: {resp.error}")

    # Start the gripper at the episode's first commanded opening (in-distribution start).
    gripper_stub.SendMotorCommand(gripper_pb2.MotorCommand(
        motor1_goal=float(actions[0, 9]), motor2_goal=float(actions[0, 10])))
    time.sleep(0.5)

    input(f"Place the object as in episode {args.episode}'s video, then press Enter to replay...")

    dt = 1.0 / args.fps
    n_fail, first_fail = 0, None
    for t in range(n):
        loop_start = time.perf_counter()
        a = actions[t]
        # Exact same calls as evaluate.py's inference loop — but CHECK the
        # response: the server reports IK failures / workspace-limit rejections
        # per command, which is how a replay "stops early" (arm freezes while
        # deltas keep integrating toward an unreachable target).
        resp = arm_stub.SendCartesianDelta(arm_pb2.CartesianDelta(
            dx=float(a[0]), dy=float(a[1]), dz=float(a[2]), dr6d=a[3:9].tolist()))
        if not resp.success:
            n_fail += 1
            if first_fail is None:
                first_fail = t
                logger.warning(f"  step {t}: arm REJECTED delta: {resp.error}")
        gripper_stub.SendMotorCommand(gripper_pb2.MotorCommand(
            motor1_goal=float(a[9]), motor2_goal=float(a[10])))
        if t % 25 == 0:
            logger.info(f"  step {t:4d}/{n}  |Δpos| {np.linalg.norm(a[:3]) * 1000:5.1f} mm  "
                        f"gripper ({a[9]:+.2f}, {a[10]:+.2f})"
                        + (f"  [{n_fail} rejected]" if n_fail else ""))
        time.sleep(max(0.0, dt - (time.perf_counter() - loop_start)))

    if n_fail:
        logger.warning(f"Arm rejected {n_fail}/{n} deltas (first at step {first_fail}). "
                       f"The demo motion exits the arm's reachable envelope from this start "
                       f"pose (joint limits / workspace) — try a different start pose, or "
                       f"accept that this demo's full extent isn't reachable.")
    logger.info("Replay done. Same motion shape as the demo video → frames agree. "
                "Consistently skewed direction → control-frame/axis mismatch.")


if __name__ == "__main__":
    main()
