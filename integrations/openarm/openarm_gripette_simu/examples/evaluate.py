"""Evaluate a trained Diffusion Policy on the Gripette simulator over multiple episodes.

Runs repeated episodes with environment reset and randomization.
Auto-detects the observation.state mode (2D gripper-only or 11D relative proprioception)
from the checkpoint.

Usage:
  uv run python examples/evaluate.py \\
      --checkpoint outputs/gripette/diffusion \\
      --num_episodes 20

  # With debug visualization:
  uv run python examples/evaluate.py \\
      --checkpoint outputs/gripette/diffusion \\
      --num_episodes 5 --debug
"""

import argparse
import logging
import time

import cv2
import grpc
import numpy as np
import torch

from lerobot.policies.factory import make_pre_post_processors
import json as _json
from pathlib import Path as _Path

from lerobot.policies.factory import get_policy_class


def _load_policy_any(checkpoint: str):
    """Load any LeRobot policy from a checkpoint, dispatching on the `type`
    field in its config.json (e.g. 'diffusion', 'act', 'pi0_fast'). This
    replaces the previous hardcoded DiffusionPolicy.from_pretrained so the
    same eval works for the Diffusion / ACT / Pi0Fast comparison arms.

    Falls back to a local config.json read; for Hub repos the file is fetched.
    """
    cfg_path = _Path(checkpoint) / "config.json"
    if cfg_path.is_file():
        policy_type = _json.loads(cfg_path.read_text())["type"]
    else:
        from huggingface_hub import hf_hub_download

        policy_type = _json.loads(
            _Path(hf_hub_download(checkpoint, "config.json")).read_text()
        )["type"]
    return get_policy_class(policy_type).from_pretrained(checkpoint)
from scipy.spatial.transform import Rotation
from openarm_gripette_simu.rotation import (
    rotation_6d_to_matrix as rotation_6d_to_rotation_matrix_numpy,
    rotation_matrix_to_6d as rotation_matrix_to_rotation_6d_numpy,
)


def clamp_delta(delta_pos, delta_rot_6d, clamp_pos_m, clamp_rot_rad):
    """Clip a Cartesian-delta action's magnitude (safety test for outlier samples).

    Diffusion samples from the learned action distribution; on the wide v9
    distribution it occasionally draws an outlier delta that drives the
    integrator into a near-singular pose ("explosion"). Clamping the per-step
    position-delta norm and rotation-delta angle caps those outliers. Returns
    (delta_pos, delta_rot_6d, was_clamped).
    """
    was = False
    if clamp_pos_m is not None:
        n = float(np.linalg.norm(delta_pos))
        if n > clamp_pos_m:
            delta_pos = delta_pos * (clamp_pos_m / n)
            was = True
    if clamp_rot_rad is not None:
        R = rotation_6d_to_rotation_matrix_numpy(delta_rot_6d.reshape(1, 6))[0]
        rotvec = Rotation.from_matrix(R).as_rotvec()
        ang = float(np.linalg.norm(rotvec))
        if ang > clamp_rot_rad:
            R = Rotation.from_rotvec(rotvec * (clamp_rot_rad / ang)).as_matrix()
            delta_rot_6d = rotation_matrix_to_rotation_6d_numpy(R.reshape(1, 3, 3))[0]
            was = True
    return delta_pos, delta_rot_6d, was

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Gripette policy on simulator")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    p.add_argument("--arm_addr", type=str, default="localhost:50052", help="ArmService gRPC address")
    p.add_argument("--gripper_addr", type=str, default="localhost:50051", help="GripperService gRPC address")
    p.add_argument("--device", type=str, default="cuda", help="Compute device")
    p.add_argument("--num_episodes", type=int, default=20, help="Number of evaluation episodes")
    p.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    p.add_argument("--fps", type=float, default=10.0, help="Control loop frequency")
    p.add_argument("--clamp_pos_mm", type=float, default=None,
                   help="Safety test: clip per-step Cartesian position-delta norm to this (mm). "
                        "Caps outlier samples (e.g. Diffusion 'explosions'). Cartesian only.")
    p.add_argument("--clamp_rot_deg", type=float, default=None,
                   help="Safety test: clip per-step rotation-delta angle to this (deg). Cartesian only.")
    p.add_argument("--success_check_freq", type=int, default=10, help="Check success every N steps")
    p.add_argument("--debug", action="store_true", help="Show camera feed during evaluation")
    p.add_argument("--log_gripper", action="store_true",
                   help="Print the gripper command (proximal/distal) sent each step, vs the observed gripper state")
    p.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Override the checkpoint's n_action_steps at inference (re-planning "
        "cadence). 1 = re-infer every step (tightest closed loop). Lower values "
        "help policies that drift open-loop (notably ACT). None = use checkpoint value.",
    )
    p.add_argument(
        "--task",
        type=str,
        default="grasp and lift cube",
        help="Language task string for VLA policies (Pi0/Pi0Fast/Pi0.5). Ignored "
        "by Diffusion/ACT. Should match (cleaned) the task used at training time "
        "— the dataset's task was 'grasp_and_lift_cube', which the Pi0Fast "
        "processor cleans to 'grasp and lift cube'.",
    )
    return p.parse_args()


def get_camera_frame(gripper_stub, gripper_pb2):
    """Get latest camera frame and gripper state from the streaming service."""
    for frame in gripper_stub.StreamState(gripper_pb2.StreamRequest()):
        img_bgr = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gripper = np.array(
            [frame.motor_state.motor1_position, frame.motor_state.motor2_position],
            dtype=np.float32,
        )
        return img_rgb, gripper
    raise RuntimeError("No frame received from camera stream")


def capture_start_pose(arm_stub, arm_pb2):
    """Capture the EE pose at episode start for relative proprioception."""
    arm_state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
    start_pos = np.array([arm_state.x, arm_state.y, arm_state.z], dtype=np.float32)
    start_r6d = np.array(list(arm_state.r6d), dtype=np.float32)
    start_rot = rotation_6d_to_rotation_matrix_numpy(start_r6d.reshape(1, 6))[0]
    return start_pos, start_rot


def compute_relative_state(arm_state, gripper_joints, start_pos, start_rot):
    """Compute 11D relative state: [pos_rel(3), rot_rel_6d(6), gripper(2)]."""
    pos = np.array([arm_state.x, arm_state.y, arm_state.z], dtype=np.float32)
    rot_6d = np.array(list(arm_state.r6d), dtype=np.float32)
    r_current = rotation_6d_to_rotation_matrix_numpy(rot_6d.reshape(1, 6))[0]

    # Pose relative to start, in the START camera frame (gripper-egocentric /
    # frame-independent — MUST match convert_dataset.py):
    #   rel_pos = R_start^T @ (pos - start_pos);  R_rel = R_start^T @ R_current
    rel_pos = start_rot.T @ (pos - start_pos)
    r_relative = start_rot.T @ r_current
    rel_rot_6d = rotation_matrix_to_rotation_6d_numpy(r_relative.reshape(1, 3, 3))[0]

    return np.concatenate([rel_pos, rel_rot_6d, gripper_joints])


def build_observation(
    arm_stub,
    arm_pb2,
    gripper_stub,
    gripper_pb2,
    use_relative_proprio,
    start_pos,
    start_rot,
    joint_mode=False,
):
    """Build the full observation (camera image + state) for one step."""
    camera_image, gripper_joints = get_camera_frame(gripper_stub, gripper_pb2)

    if joint_mode:
        # Joint-space state = [arm_q(7), proximal, distal], matching
        # convert_to_jointspace.py. arm_q from GetArmState; gripper from the
        # gripper service (same 2D as the Cartesian path).
        arm_state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
        state = np.concatenate(
            [np.array(arm_state.joint_positions, dtype=np.float32), gripper_joints]
        )
    elif use_relative_proprio:
        arm_state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
        state = compute_relative_state(arm_state, gripper_joints, start_pos, start_rot)
    else:
        state = gripper_joints

    return camera_image, state


def run_episode(
    policy,
    preprocessor,
    postprocessor,
    arm_stub,
    gripper_stub,
    arm_pb2,
    gripper_pb2,
    device,
    max_steps,
    fps,
    success_check_freq,
    debug,
    use_relative_proprio,
    start_pos,
    start_rot,
    task,
    joint_mode=False,
    clamp_pos_m=None,
    clamp_rot_rad=None,
    log_gripper=False,
) -> dict:
    """Run a single evaluation episode. Returns dict with stats."""
    n_clamped = 0
    dt = 1.0 / fps
    episode_start = time.perf_counter()

    for step in range(max_steps):
        loop_start = time.perf_counter()

        # --- Observe ---
        camera_image, state = build_observation(
            arm_stub,
            arm_pb2,
            gripper_stub,
            gripper_pb2,
            use_relative_proprio,
            start_pos,
            start_rot,
            joint_mode=joint_mode,
        )

        state_tensor = torch.from_numpy(state).float()
        image_tensor = torch.from_numpy(camera_image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()

        batch = {
            "observation.state": state_tensor.unsqueeze(0).to(device),
            "observation.images.cam0": image_tensor.unsqueeze(0).to(device),
            # VLA policies (Pi0/Pi0Fast/Pi0.5) require a language task string;
            # their preprocessor tokenizes it into the prompt. Classic policies
            # (Diffusion/ACT) ignore it — exactly as during training, where the
            # dataset always carried a `task` field. Harmless to always include.
            "task": task,
        }

        # --- Inference ---
        batch = preprocessor(batch)
        with torch.no_grad():
            action = policy.select_action(batch)
        action = postprocessor(action)

        action_np = action.squeeze(0).cpu().numpy()

        # --- Send commands ---
        if joint_mode:
            # 9D joint action: [arm_q(7), proximal, distal]. Arm joints go
            # straight to the arm (no integrator/IK); gripper via the gripper
            # service exactly as in the Cartesian path.
            arm_joints = action_np[:7]
            gripper_goal = action_np[7:9]
            arm_stub.SendJointCommand(arm_pb2.JointCommand(joint_positions=arm_joints.tolist()))
            delta_pos = None
        else:
            delta_pos = action_np[:3]
            delta_rot_6d = action_np[3:9]
            gripper_goal = action_np[9:]
            if clamp_pos_m is not None or clamp_rot_rad is not None:
                delta_pos, delta_rot_6d, was = clamp_delta(
                    delta_pos, delta_rot_6d, clamp_pos_m, clamp_rot_rad)
                n_clamped += int(was)
            arm_stub.SendCartesianDelta(
                arm_pb2.CartesianDelta(
                    dx=float(delta_pos[0]),
                    dy=float(delta_pos[1]),
                    dz=float(delta_pos[2]),
                    dr6d=delta_rot_6d.tolist(),
                )
            )
        gripper_stub.SendMotorCommand(
            gripper_pb2.MotorCommand(
                motor1_goal=float(gripper_goal[0]),
                motor2_goal=float(gripper_goal[1]) if len(gripper_goal) > 1 else 0.0,
            )
        )

        if log_gripper:
            # state[-2:] is always the observed gripper (2D-only, relative, and
            # joint-space states all end with [proximal, distal]).
            obs_g = state[-2:]
            cmd_dist = gripper_goal[1] if len(gripper_goal) > 1 else 0.0
            obs_dist = obs_g[1] if len(obs_g) > 1 else 0.0
            print(
                f"step {step:3d} | gripper cmd: prox={gripper_goal[0]:+.4f} dist={cmd_dist:+.4f}"
                f" | obs: prox={obs_g[0]:+.4f} dist={obs_dist:+.4f}",
                flush=True,
            )

        # --- Debug display ---
        if debug:
            img_display = camera_image.copy()
            label = (f"Step {step} | joint cmd" if joint_mode
                     else f"Step {step} | delta {np.linalg.norm(delta_pos) * 1000:.1f}mm")
            cv2.putText(
                img_display,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Evaluation", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        # --- Check success ---
        if step > 0 and step % success_check_freq == 0:
            status = arm_stub.GetSuccessStatus(arm_pb2.SuccessStatusRequest())
            if status.goal_reached:
                return {
                    "success": True,
                    "steps": step + 1,
                    "displacement_mm": status.cube_displacement * 1000,
                    "duration_s": time.perf_counter() - episode_start,
                    "n_clamped": n_clamped,
                }

        # --- Timing ---
        elapsed = time.perf_counter() - loop_start
        if (remaining := dt - elapsed) > 0:
            time.sleep(remaining)

    # Episode ended without success
    status = arm_stub.GetSuccessStatus(arm_pb2.SuccessStatusRequest())
    return {
        "success": status.goal_reached,
        "steps": max_steps,
        "displacement_mm": status.cube_displacement * 1000,
        "duration_s": time.perf_counter() - episode_start,
        "n_clamped": n_clamped,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)

    # ---- Load policy (any type: diffusion / act / pi0_fast / ...) ----
    logger.info(f"Loading policy from {args.checkpoint}")
    policy = _load_policy_any(args.checkpoint)
    logger.info(f"Loaded policy type: {policy.config.type}")
    # Optional re-planning-cadence override. Smaller n_action_steps re-infers
    # more often (tighter closed loop), which matters a lot for policies that
    # drift off-distribution during open-loop chunk execution — ACT in
    # particular is designed for very frequent re-planning / temporal
    # ensembling, and executing long chunks (its trained default of 8+) can
    # cause it to wander off the grasp manifold and never trigger the close.
    if args.n_action_steps is not None:
        logger.info(
            f"Overriding n_action_steps: {policy.config.n_action_steps} -> {args.n_action_steps}"
        )
        policy.config.n_action_steps = args.n_action_steps
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)

    # Auto-detect state/action mode from the policy's feature shapes.
    state_dim = policy.config.robot_state_feature.shape[0]
    action_dim = policy.config.action_feature.shape[0]
    joint_mode = action_dim == 9  # 9D = [arm_q(7), prox, dist]; 11D = Cartesian deltas
    use_relative_proprio = (state_dim > 2) and not joint_mode
    logger.info(
        f"Policy: action_space={'joint' if joint_mode else 'cartesian'}, "
        f"state_dim={state_dim}, action_dim={action_dim}, "
        f"n_action_steps={policy.config.n_action_steps}"
    )

    # ---- Connect to simulator ----
    from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc, gripper_pb2, gripper_pb2_grpc

    arm_channel = grpc.insecure_channel(args.arm_addr)
    arm_stub = arm_pb2_grpc.ArmServiceStub(arm_channel)
    gripper_channel = grpc.insecure_channel(args.gripper_addr)
    gripper_stub = gripper_pb2_grpc.GripperServiceStub(gripper_channel)

    arm_stub.Ping(arm_pb2.ArmPingRequest())
    gripper_stub.Ping(gripper_pb2.PingRequest())
    logger.info("Connected to simulator")

    # ---- Evaluation loop ----
    results = []
    logger.info(
        f"\nStarting evaluation: {args.num_episodes} episodes, "
        f"max {args.max_steps} steps/episode at {args.fps} Hz\n"
    )

    for ep in range(args.num_episodes):
        # Reset environment with randomization
        reset_resp = arm_stub.Reset(arm_pb2.ResetRequest())
        if not reset_resp.success:
            logger.error(f"Reset failed: {reset_resp.error}")
            continue

        # Explicitly re-open the gripper between episodes. arm_stub.Reset()
        # only re-randomizes the arm/cube; without this, the gripper retains
        # the closed state from the previous episode's grasp + lift, which
        # is severely out of distribution for the policy at episode start.
        gripper_stub.SendMotorCommand(gripper_pb2.MotorCommand(motor1_goal=0.0, motor2_goal=0.0))

        # Reset policy action queue
        policy.reset()

        # Small delay for physics to settle (and for gripper to actually open)
        time.sleep(0.5)

        # Capture start pose for relative proprioception (after reset)
        start_pos, start_rot = None, None
        if use_relative_proprio:
            start_pos, start_rot = capture_start_pose(arm_stub, arm_pb2)

        logger.info(
            f"Episode {ep + 1}/{args.num_episodes} — "
            f"cube at ({reset_resp.cube_x:.3f}, {reset_resp.cube_y:.3f}, {reset_resp.cube_z:.3f})"
        )

        result = run_episode(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            arm_stub=arm_stub,
            gripper_stub=gripper_stub,
            arm_pb2=arm_pb2,
            gripper_pb2=gripper_pb2,
            device=device,
            max_steps=args.max_steps,
            fps=args.fps,
            success_check_freq=args.success_check_freq,
            debug=args.debug,
            log_gripper=args.log_gripper,
            use_relative_proprio=use_relative_proprio,
            start_pos=start_pos,
            start_rot=start_rot,
            task=args.task,
            joint_mode=joint_mode,
            clamp_pos_m=(args.clamp_pos_mm / 1000.0) if args.clamp_pos_mm else None,
            clamp_rot_rad=(np.deg2rad(args.clamp_rot_deg)) if args.clamp_rot_deg else None,
        )
        results.append(result)

        status_str = "SUCCESS" if result["success"] else "FAIL"
        logger.info(
            f"  -> {status_str} | steps: {result['steps']:>3d} | "
            f"displacement: {result['displacement_mm']:.1f}mm | "
            f"time: {result['duration_s']:.1f}s"
        )

    # ---- Summary ----
    num_success = sum(r["success"] for r in results)
    num_total = len(results)
    success_rate = num_success / num_total * 100 if num_total > 0 else 0
    avg_displacement = np.mean([r["displacement_mm"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    success_results = [r for r in results if r["success"]]
    avg_success_steps = np.mean([r["steps"] for r in success_results]) if success_results else 0

    print(f"\n{'=' * 60}")
    print("  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  State mode:       {'relative proprio (11D)' if use_relative_proprio else 'gripper only (2D)'}")
    print(f"  Episodes:         {num_total}")
    print(f"  Success rate:     {num_success}/{num_total} ({success_rate:.1f}%)")
    print(f"  Avg displacement: {avg_displacement:.1f} mm")
    print(f"  Avg steps (all):  {avg_steps:.0f}")
    if success_results:
        print(f"  Avg steps (success): {avg_success_steps:.0f}")
    if args.clamp_pos_mm or args.clamp_rot_deg:
        total_clamped = sum(r.get("n_clamped", 0) for r in results)
        print(f"  Action clamp:     pos<={args.clamp_pos_mm}mm rot<={args.clamp_rot_deg}deg "
              f"({total_clamped} steps clamped across {num_total} eps)")
    print(f"{'=' * 60}")

    if args.debug:
        cv2.destroyAllWindows()
    arm_channel.close()
    gripper_channel.close()


if __name__ == "__main__":
    main()
