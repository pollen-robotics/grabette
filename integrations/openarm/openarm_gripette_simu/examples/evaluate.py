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
import threading
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
    p.add_argument("--max_ticks", type=int, default=1,
                   help="Max chunk actions consumed per loop iteration to compensate a "
                        "slow loop (wallclock catch-up). 1 (default) = one action per "
                        "iteration: smooth motion at whatever fraction of demo speed the "
                        "loop achieves. >1 only makes sense if --log_latency shows the "
                        "loop USUALLY holds the target rate: each tick costs ~20 ms "
                        "(send incl. server IK + amortized inference), so a chronically "
                        "slow loop just saturates the cap and moves in violent bursts.")
    p.add_argument("--async_exec", action="store_true",
                   help="Async execution: a sender thread streams actions to the arm at "
                        "EXACTLY --fps (the demo clock) while inference replans in "
                        "parallel from the freshest camera pair — reproducing the "
                        "training/sim dynamics (demo-speed motion, fresh feedback, no "
                        "pauses). Use --fps 50 to match a 50fps dataset. Gripper-only "
                        "(2D state) cartesian models only; ignores --max_ticks/"
                        "--skip_stale (both are subsumed).")
    p.add_argument("--skip_stale", action="store_true",
                   help="Latency compensation (UMI-style): at each replan, discard the "
                        "chunk-head actions corresponding to motion the arm already "
                        "executed while the observation frame aged (k = frame staleness "
                        "/ loop period). Counters the systematic overshoot ('push through "
                        "the object') caused by planning from a 100-300ms-old frame. "
                        "Diffusion policies only (needs the action queue).")
    p.add_argument("--success_check_freq", type=int, default=10, help="Check success every N steps")
    p.add_argument("--debug", action="store_true", help="Show camera feed during evaluation")
    p.add_argument("--log_gripper", action="store_true",
                   help="Print the gripper command (proximal/distal) sent each step, vs the observed gripper state")
    p.add_argument("--log_deltas", action="store_true",
                   help="Print the exact Cartesian delta sent to the arm each step "
                        "(post-clamp): Δpos per axis + magnitude (mm), rotation-delta "
                        "angle (deg), and the gripper goals.")
    p.add_argument("--ask_success", type=str, default=None, metavar="RESULTS_JSONL",
                   help="REAL-ARM scoring: after each episode, prompt the operator for "
                        "grasp success (y/N) and append {episode, success, steps, "
                        "checkpoint, ...} to this JSONL. Use for A/B sessions — the sim's "
                        "automatic success check is a stub on the real server.")
    p.add_argument("--log_latency", action="store_true",
                   help="Measure the perception→action latency chain each step: true camera "
                        "rate + stale-frame detection (inter-frame server timestamps), frame "
                        "staleness above best-case (buffering), and inference time. Prints a "
                        "per-episode summary. Training data assumes ZERO obs→act lag — if the "
                        "measured lag spans several control periods, the policy is acting on "
                        "the past (symptoms: jerky/oscillating endgame, failed fine alignment).")
    p.add_argument("--dump_obs", type=str, default=None,
                   help="Directory to dump the EXACT observations fed to the policy "
                        "(obs_XXXXX.png + state.jsonl, one subdir per episode). Use with "
                        "--num_episodes 1 for a train/deploy distribution check (ood_check.py).")
    p.add_argument("--start_gripper", type=float, nargs=2, default=[0.0, 0.0],
                   metavar=("PROX", "DIST"),
                   help="Gripper opening commanded at each episode start. MUST match the "
                        "demos' typical first-frame state, or the policy starts conditioned "
                        "on an out-of-distribution gripper state. Sim datasets start fully "
                        "open (0 0, the default); real Grabette demos start partially "
                        "squeezed (e.g. 0.40 0.30 for the pick-can dataset — check with "
                        "ood_check.py / the dataset's first-frame stats).")
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


class CameraStream:
    """Persistent camera stream: one background thread keeps StreamState open
    and always holds the LATEST decoded frame.

    Why: the previous per-step pattern (open a fresh gRPC stream, block for its
    next emission) cost 180-330 ms per observation — measured to silently run
    the whole control loop at ~5 Hz instead of 50, executing the policy in
    slow-motion with target hops (jerky arm, failed fine alignment).

    get() returns (img_rgb, gripper, frame_ts_ms) instantly. frame_ts_ms is the
    SERVER's monotonic capture timestamp: not comparable to the local clock,
    but inter-frame deltas give the true camera rate, and an unchanged value
    marks a stale (already-consumed) frame.
    """

    def __init__(self, gripper_stub, gripper_pb2):
        self._stub = gripper_stub
        self._pb2 = gripper_pb2
        self._lock = threading.Lock()
        self._latest = None
        # Short history of recent frames (newest last) so consumers can build
        # a 2-observation pair (the policy is n_obs_steps=2: it conditions on
        # inter-frame motion). 8 frames ≈ 0.4 s at 20 Hz.
        self._history = []
        self._ready = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop:
            try:
                for frame in self._stub.StreamState(self._pb2.StreamRequest()):
                    if self._stop:
                        return
                    img_bgr = cv2.imdecode(
                        np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    gripper = np.array(
                        [frame.motor_state.motor1_position,
                         frame.motor_state.motor2_position],
                        dtype=np.float32,
                    )
                    with self._lock:
                        self._latest = (img_rgb, gripper, float(frame.timestamp_ms))
                        self._history.append(self._latest)
                        if len(self._history) > 8:
                            self._history.pop(0)
                    self._ready.set()
            except Exception as e:  # noqa: BLE001 — stream drop: reconnect
                if not self._stop:
                    logger.warning(f"Camera stream dropped ({e}); reconnecting...")
                    time.sleep(0.2)

    def get(self, timeout: float = 5.0):
        if not self._ready.wait(timeout):
            raise RuntimeError("No frame received from camera stream")
        with self._lock:
            return self._latest

    def get_pair(self, timeout: float = 5.0):
        """Return (previous_frame, latest_frame) — the two most recent DISTINCT
        camera frames, each (img_rgb, gripper, ts_ms). The closest available
        approximation of the 20 ms-spaced observation pair the policy was
        trained on (camera period sets the floor). Duplicates latest if only
        one frame exists yet."""
        if not self._ready.wait(timeout):
            raise RuntimeError("No frame received from camera stream")
        with self._lock:
            now = self._history[-1]
            prev = self._history[-2] if len(self._history) >= 2 else now
            return prev, now

    def stop(self):
        self._stop = True


class ChunkExecutor:
    """Async executor: streams per-tick actions to the arm at EXACTLY `fps`
    from a replaceable chunk, while inference replans in parallel.

    Why: the training data is 50 fps and one action = one 20 ms tick. A
    synchronous observe-infer-act loop can never hold 50 Hz (inference alone
    is ~80 ms), so motion runs at a fraction of demo speed and the policy
    sees dynamics it was never trained on. Here the sender thread paces the
    demo clock; submit() swaps in a fresher chunk whenever one is ready
    (receding horizon), skipping the chunk-head actions that duplicate motion
    already executed since the observation was captured.
    """

    def __init__(self, arm_stub, arm_pb2, gripper_stub, gripper_pb2, fps,
                 clamp_pos_m=None, clamp_rot_rad=None,
                 start_pos=None, start_rot=None):
        self._arm_stub = arm_stub
        self._arm_pb2 = arm_pb2
        self._gripper_stub = gripper_stub
        self._gripper_pb2 = gripper_pb2
        self._dt = 1.0 / fps
        self._clamp_pos_m = clamp_pos_m
        self._clamp_rot_rad = clamp_rot_rad
        self._lock = threading.Lock()
        self._chunk = []
        self._i = 0
        # Integrated COMMANDED pose (world frame), seeded from the start FK:
        # p += R @ dp ; R = R @ R_delta for every ACCEPTED delta — what the
        # arm SHOULD have done. Compared to measured FK it quantifies
        # tracking gain/lag/overshoot (telemetry, read via cmd_pose()).
        self._p_cmd = start_pos.copy() if start_pos is not None else None
        self._R_cmd = start_rot.copy() if start_rot is not None else None
        # Counters (int reads/writes are atomic under the GIL).
        self.sent_count = 0   # ticks consumed — the executor's clock
        self.underruns = 0    # ticks with no action available (inference late)
        self.n_rejected = 0
        self.n_clamped = 0
        self.frozen = None    # set to the error string on watchdog latch
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, chunk, skip):
        """Replace the current chunk. `skip` = ticks executed since the obs
        that generated this chunk was captured — those actions describe motion
        already done, so they are dropped (always keep at least one action).
        Returns the number actually skipped."""
        with self._lock:
            k = int(np.clip(skip, 0, max(len(chunk) - 1, 0)))
            self._chunk = chunk[k:]
            self._i = 0
        return k

    def _run(self):
        next_t = time.monotonic()
        while not self._stop:
            a = None
            with self._lock:
                if self._i < len(self._chunk):
                    a = self._chunk[self._i]
                    self._i += 1
            if a is None:
                self.underruns += 1
            else:
                dp, dr6, grip = a[:3], a[3:9], a[9:]
                if self._clamp_pos_m is not None or self._clamp_rot_rad is not None:
                    dp, dr6, was = clamp_delta(dp, dr6, self._clamp_pos_m, self._clamp_rot_rad)
                    self.n_clamped += int(was)
                try:
                    resp = self._arm_stub.SendCartesianDelta(
                        self._arm_pb2.CartesianDelta(
                            dx=float(dp[0]), dy=float(dp[1]), dz=float(dp[2]),
                            dr6d=dr6.tolist(),
                        )
                    )
                    if not resp.success:
                        self.n_rejected += 1
                        if "frozen" in resp.error:
                            self.frozen = resp.error
                            return
                    elif self._p_cmd is not None:
                        R_i = rotation_6d_to_rotation_matrix_numpy(np.asarray(dr6).reshape(1, 6))[0]
                        with self._lock:
                            self._p_cmd = self._p_cmd + self._R_cmd @ dp
                            self._R_cmd = self._R_cmd @ R_i
                    # Gripper: fire-and-forget future — a blocking round trip
                    # to the Pi over WiFi would eat the 20 ms tick budget.
                    self._gripper_stub.SendMotorCommand.future(
                        self._gripper_pb2.MotorCommand(
                            motor1_goal=float(grip[0]),
                            motor2_goal=float(grip[1]) if len(grip) > 1 else 0.0,
                        )
                    )
                except Exception as e:  # noqa: BLE001 — surface, don't die silently
                    logger.warning(f"Executor send failed: {e}")
                self.sent_count += 1
            next_t += self._dt
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif sleep_for < -0.2:  # long stall (GIL/debugger) — resync, don't burst
                next_t = time.monotonic()

    def cmd_pose(self):
        with self._lock:
            return (None, None) if self._p_cmd is None else (self._p_cmd.copy(), self._R_cmd.copy())

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2.0)


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
    camera,
    use_relative_proprio,
    start_pos,
    start_rot,
    joint_mode=False,
):
    """Build the full observation (camera image + state) for one step.

    Returns (camera_image, state, frame_ts_ms)."""
    camera_image, gripper_joints, frame_ts_ms = camera.get()

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

    return camera_image, state, frame_ts_ms


def run_episode(
    policy,
    preprocessor,
    postprocessor,
    arm_stub,
    gripper_stub,
    arm_pb2,
    gripper_pb2,
    camera,
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
    max_ticks=1,
    skip_stale=False,
    log_gripper=False,
    log_deltas=False,
    log_latency=False,
    dump_dir=None,
) -> dict:
    """Run a single evaluation episode. Returns dict with stats."""
    n_clamped = 0
    n_rejected = 0  # arm commands refused by the server (IK-jump watchdog etc.)
    dt = 1.0 / fps
    episode_start = time.perf_counter()
    if dump_dir is not None:
        dump_dir = _Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
    # Latency bookkeeping. frame_ts is the SERVER's monotonic clock, so an
    # absolute frame→local age is unknowable; instead track (a) inter-frame
    # timestamp deltas = true camera rate + stale-duplicate detection, and
    # (b) staleness = (local_recv − frame_ts) above the episode's minimum
    # offset — how much OLDER than best-case each frame is (buffering/queuing).
    lat_prev_ts = None
    lat_min_offset = float("inf")
    lat_stats = {"dts": [], "stale": 0, "staleness": [], "infer": [], "ticks": []}
    # Wallclock-consistent execution state: t_nominal tracks how much demo
    # time has been sent to the arm; each iteration covers the elapsed real
    # time, capped at max_ticks. CAUTION — the cap only helps when the loop
    # is USUALLY at 50 Hz with occasional hiccups: each tick costs ~20 ms
    # (one send incl. server-side IK + amortized inference), so if the loop
    # can't hold rate the compensation saturates at the cap and delivers
    # violent multi-tick motion bursts instead of catching up. max_ticks=1
    # disables it: smooth motion at whatever fraction of demo speed the
    # loop achieves.
    t_nominal = None
    # Loop-period history for --skip_stale: converts the measured frame
    # staleness (wall ms) into "how many chunk actions the arm has already
    # executed since this observation was captured".
    prev_loop_start = None
    loop_periods = []

    for step in range(max_steps):
        loop_start = time.perf_counter()
        if prev_loop_start is not None:
            loop_periods.append(loop_start - prev_loop_start)
        prev_loop_start = loop_start

        # --- Observe ---
        camera_image, state, frame_ts_ms = build_observation(
            arm_stub,
            arm_pb2,
            camera,
            use_relative_proprio,
            start_pos,
            start_rot,
            joint_mode=joint_mode,
        )
        recv_ms = time.perf_counter() * 1000.0
        lat_min_offset = min(lat_min_offset, recv_ms - frame_ts_ms)
        frame_staleness = (recv_ms - frame_ts_ms) - lat_min_offset
        frame_dts = (frame_ts_ms - lat_prev_ts) if lat_prev_ts is not None else None
        lat_prev_ts = frame_ts_ms

        # Dump the exact observation fed to the policy (pre-normalization), for
        # offline train/deploy distribution checks (DiffusionPolicy/ood_check.py).
        # camera_image is RGB HWC uint8 here; cv2.imwrite expects BGR.
        if dump_dir is not None:
            # Lowest PNG compression: default (3) costs tens of ms per full-res
            # frame, a large fraction of the 20 ms loop budget. Still lossless.
            cv2.imwrite(str(dump_dir / f"obs_{step:05d}.png"),
                        cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 1])
            with open(dump_dir / "state.jsonl", "a") as f:
                f.write(_json.dumps({"step": step, "state": [float(v) for v in state]}) + "\n")

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

        # --- Inference + wallclock-consistent execution ---
        # The training data is 50 fps: one action = the motion of ONE 20 ms
        # tick. If the loop can't hold 50 Hz (camera acquisition, inference),
        # sending one delta per iteration executes the demo in slow motion
        # with target hops (the measured 5 Hz jerky-and-10x-slow failure).
        # Instead, consume as many chunk actions as WALL-CLOCK ticks elapsed
        # since the last iteration and send each as its own command (a replay
        # burst) — motion runs at demonstrated speed whatever the loop rate.
        t_inf = time.perf_counter()
        batch = preprocessor(batch)

        if joint_mode:
            with torch.no_grad():
                action = policy.select_action(batch)
            action = postprocessor(action)
            infer_ms = (time.perf_counter() - t_inf) * 1000.0
            action_np = action.squeeze(0).cpu().numpy()
            # 9D joint action: [arm_q(7), proximal, distal] — absolute, no
            # composition needed; slow loops just track a slower reference.
            arm_joints = action_np[:7]
            gripper_goal = action_np[7:9]
            arm_stub.SendJointCommand(arm_pb2.JointCommand(joint_positions=arm_joints.tolist()))
            delta_pos, n_ticks = None, 1
        else:
            now = time.perf_counter()
            if t_nominal is None:
                t_nominal = now
            n_ticks = int(np.clip(round((now - t_nominal) / dt), 1, max_ticks))
            t_nominal += n_ticks * dt
            if abs(now - t_nominal) > 0.5:  # lost sync (pause/debugger) → resync
                t_nominal = now

            # Send each tick's delta as its OWN command, exactly like a
            # training-rate replay burst. The arm integrator accumulates them
            # identically to one composed delta, but each command stays small
            # (one tick + client clamps), which is what the server's IK-jump
            # watchdog is calibrated for: a single COMPOUND delta of n_ticks
            # motion looks like a singularity branch flip (>15deg on one joint
            # in one command) and latches the arm frozen. p_acc/R_acc compose
            # the ACCEPTED deltas for logging only.
            p_acc = np.zeros(3)
            R_acc = np.eye(3)
            gripper_goal = None
            frozen_error = None
            for _ in range(n_ticks):
                # Latency compensation (UMI-style): when the policy is about
                # to REPLAN (action queue empty), the new chunk starts from
                # the pose in the observation — but that frame is stale, and
                # the arm has kept executing while it aged. The chunk's first
                # k actions describe motion the arm has ALREADY done; executing
                # them again overshoots (the "push through the object" failure).
                # Discard them: k = staleness / measured loop period.
                if skip_stale and loop_periods:
                    q = getattr(policy, "_queues", None)
                    q = q.get("action") if isinstance(q, dict) else None
                    if q is not None and len(q) == 0:
                        period_ms = 1000.0 * float(np.median(loop_periods[-20:]))
                        k = int(np.clip(round(frame_staleness / max(period_ms, 1.0)),
                                        0, policy.config.n_action_steps - 1))
                        for _ in range(k):
                            with torch.no_grad():
                                policy.select_action(batch)  # discard stale head
                        if log_latency and k:
                            print(f"  skip_stale: dropped {k} chunk-head action(s) "
                                  f"(staleness {frame_staleness:.0f}ms / period {period_ms:.0f}ms)",
                                  flush=True)
                with torch.no_grad():
                    a = policy.select_action(batch)
                a = postprocessor(a)
                a_np = a.squeeze(0).cpu().numpy()
                dp, dr6, gripper_goal = a_np[:3], a_np[3:9], a_np[9:]
                if clamp_pos_m is not None or clamp_rot_rad is not None:
                    dp, dr6, was = clamp_delta(dp, dr6, clamp_pos_m, clamp_rot_rad)
                    n_clamped += int(was)
                resp = arm_stub.SendCartesianDelta(
                    arm_pb2.CartesianDelta(
                        dx=float(dp[0]), dy=float(dp[1]), dz=float(dp[2]),
                        dr6d=dr6.tolist(),
                    )
                )
                if not resp.success:
                    # The real-arm server rejects unsafe commands (IK-jump
                    # watchdog) and, once latched, freezes ALL motion until
                    # Reset. Silently dropping these responses means running
                    # the policy against a frozen arm — surface them loudly.
                    n_rejected += 1
                    if "frozen" in resp.error:
                        frozen_error = resp.error
                        break
                    if n_rejected <= 5 or n_rejected % 25 == 0:
                        print(f"ARM REJECTED delta #{n_rejected}: {resp.error}", flush=True)
                    continue  # rejected delta was NOT applied: skip composition
                R_i = rotation_6d_to_rotation_matrix_numpy(dr6.reshape(1, 6))[0]
                p_acc = p_acc + R_acc @ dp
                R_acc = R_acc @ R_i
            infer_ms = (time.perf_counter() - t_inf) * 1000.0

            if frozen_error is not None:
                print(
                    f"\nARM MOTION FROZEN at step {step}: {frozen_error}\n"
                    f"The server's IK-jump watchdog latched (check the arm "
                    f"server log for the tripping joint). Aborting episode — "
                    f"Reset/re-home the arm before the next one.",
                    flush=True,
                )
                return {
                    "success": False,
                    "steps": step + 1,
                    "displacement_mm": 0.0,
                    "duration_s": time.perf_counter() - episode_start,
                    "n_clamped": n_clamped,
                    "n_rejected": n_rejected,
                }

            delta_pos = p_acc
            delta_rot_6d = rotation_matrix_to_rotation_6d_numpy(R_acc.reshape(1, 3, 3))[0]

        if log_latency:
            lat_stats["infer"].append(infer_ms)
            lat_stats["staleness"].append(frame_staleness)
            lat_stats["ticks"].append(n_ticks)
            if frame_dts is not None:
                lat_stats["dts"].append(frame_dts)
                if frame_dts <= 0.0:
                    lat_stats["stale"] += 1
            print(
                f"lat step {step:3d} | frame Δts {frame_dts if frame_dts is not None else 0.0:6.1f}ms"
                f"{' STALE' if frame_dts is not None and frame_dts <= 0 else ''}"
                f" | staleness +{frame_staleness:5.1f}ms | infer {infer_ms:5.1f}ms"
                f" | ticks x{n_ticks} | loop target {dt * 1000:.0f}ms",
                flush=True,
            )
        gripper_stub.SendMotorCommand(
            gripper_pb2.MotorCommand(
                motor1_goal=float(gripper_goal[0]),
                motor2_goal=float(gripper_goal[1]) if len(gripper_goal) > 1 else 0.0,
            )
        )

        if log_deltas and not joint_mode:
            # The net motion commanded this iteration (post-clamp): the
            # composition of the ACCEPTED per-tick deltas, plus gripper goals.
            r_delta = rotation_6d_to_rotation_matrix_numpy(delta_rot_6d.reshape(1, 6))[0]
            ang_deg = np.degrees(np.arccos(np.clip((np.trace(r_delta) - 1.0) / 2.0, -1.0, 1.0)))
            d_mm = delta_pos * 1000.0
            print(
                f"step {step:3d} | Δpos mm: [{d_mm[0]:+6.2f} {d_mm[1]:+6.2f} {d_mm[2]:+6.2f}]"
                f" |Δ| {np.linalg.norm(d_mm):5.2f} | Δrot {ang_deg:5.2f}° | x{n_ticks} tick(s)"
                f" | grip ({gripper_goal[0]:+.3f}, "
                f"{gripper_goal[1] if len(gripper_goal) > 1 else 0.0:+.3f})",
                flush=True,
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
                    "n_rejected": n_rejected,
                }

        # --- Timing ---
        elapsed = time.perf_counter() - loop_start
        if (remaining := dt - elapsed) > 0:
            time.sleep(remaining)

    # Episode ended without success
    status = arm_stub.GetSuccessStatus(arm_pb2.SuccessStatusRequest())
    if log_latency and lat_stats["infer"]:
        dts = np.array(lat_stats["dts"]) if lat_stats["dts"] else np.array([0.0])
        st = np.array(lat_stats["staleness"])
        inf = np.array(lat_stats["infer"])
        tk = np.array(lat_stats["ticks"])
        n = len(inf)
        print(
            f"LATENCY SUMMARY ({n} steps) | camera: median Δts {np.median(dts):.1f}ms "
            f"(≈{1000.0 / max(np.median(dts), 1e-6):.0f}Hz), stale frames {lat_stats['stale']}/{n} "
            f"({100.0 * lat_stats['stale'] / n:.0f}%) | staleness p50 {np.percentile(st, 50):.0f}ms "
            f"p95 {np.percentile(st, 95):.0f}ms | infer p50 {np.percentile(inf, 50):.0f}ms "
            f"p95 {np.percentile(inf, 95):.0f}ms | ticks/send p50 {np.percentile(tk, 50):.0f} "
            f"(1 = loop holds {1.0 / dt:.0f}Hz; >1 = wallclock compensation active) | "
            f"loop target {dt * 1000:.0f}ms",
            flush=True,
        )

    if n_rejected > 0:
        print(
            f"WARNING: the arm server rejected {n_rejected} command(s) this "
            f"episode (IK-jump watchdog) — the executed motion differs from "
            f"what the policy commanded.",
            flush=True,
        )
    return {
        "success": status.goal_reached,
        "steps": max_steps,
        "displacement_mm": status.cube_displacement * 1000,
        "duration_s": time.perf_counter() - episode_start,
        "n_clamped": n_clamped,
        "n_rejected": n_rejected,
    }


def run_episode_async(
    policy,
    preprocessor,
    postprocessor,
    arm_stub,
    gripper_stub,
    arm_pb2,
    gripper_pb2,
    camera,
    device,
    max_steps,
    fps,
    success_check_freq,
    task,
    clamp_pos_m=None,
    clamp_rot_rad=None,
    log_deltas=False,
    log_latency=False,
    dump_dir=None,
) -> dict:
    """Async episode: ChunkExecutor streams actions at exactly `fps` (the demo
    clock) while this loop replans as fast as inference allows (~10 Hz), each
    time from the freshest camera pair. This reproduces the training/sim
    dynamics: demo-speed motion, ~20 ms-scale feedback, no pauses.

    Observation pairs: the policy is n_obs_steps=2 — it conditions on
    inter-frame MOTION. Each replan feeds the two most recent distinct camera
    frames (the camera period, ~50 ms, is the closest physics allows to the
    20 ms training spacing) instead of whatever two frames consecutive loop
    iterations happened to see. Chunk handoff skips the actions already
    executed since the newest frame was captured (executor tick count + frame
    staleness), so the new chunk continues from the arm's true pose.

    Supports the gripper-only (2D state) cartesian models only.
    """
    n_act = int(policy.config.n_action_steps)
    if not isinstance(getattr(policy, "_queues", None), dict) or "action" not in policy._queues:
        raise SystemExit("--async_exec needs the lerobot queue-based policy API "
                         "(Diffusion); this policy type doesn't expose _queues['action'].")
    if dump_dir is not None:
        dump_dir = _Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

    def make_batch(img, grip):
        state = torch.from_numpy(np.asarray(grip, dtype=np.float32))
        im = torch.from_numpy(img).float().div_(255.0).permute(2, 0, 1).contiguous()
        return {
            "observation.state": state.unsqueeze(0).to(device),
            "observation.images.cam0": im.unsqueeze(0).to(device),
            "task": task,
        }

    ep_start_pos, ep_start_rot = capture_start_pose(arm_stub, arm_pb2)
    executor = ChunkExecutor(arm_stub, arm_pb2, gripper_stub, gripper_pb2, fps,
                             clamp_pos_m=clamp_pos_m, clamp_rot_rad=clamp_rot_rad,
                             start_pos=ep_start_pos.astype(np.float64),
                             start_rot=ep_start_rot.astype(np.float64))
    episode_start = time.perf_counter()
    lat_min_offset = float("inf")
    stats = {"infer": [], "skip": [], "chunk": []}
    cycle = 0
    first_cycle = True
    success = False
    try:
        while executor.sent_count < max_steps and executor.frozen is None:
            (img_prev, grip_prev, _ts_prev), (img_now, grip_now, ts_now) = camera.get_pair()
            sent_at_obs = executor.sent_count
            recv_ms = time.perf_counter() * 1000.0
            lat_min_offset = min(lat_min_offset, recv_ms - ts_now)
            staleness_ms = (recv_ms - ts_now) - lat_min_offset

            t_inf = time.perf_counter()
            with torch.no_grad():
                # First call feeds obs[t-1] and pops the leftover action we
                # deliberately keep in the queue (see drain below), so it does
                # NOT trigger generation. Second call feeds obs[t]: queue now
                # empty -> generates the chunk conditioned on the (t-1, t)
                # camera pair, matching training's consecutive-frame stacking.
                _ = policy.select_action(preprocessor(make_batch(img_prev, grip_prev)))
                if first_cycle:
                    # No leftover existed: that call generated a junk chunk
                    # from a duplicated frame. Flush it so the next call
                    # regenerates from the real pair.
                    policy._queues["action"].clear()
                    first_cycle = False
                a0 = policy.select_action(preprocessor(make_batch(img_now, grip_now)))
            actions = [postprocessor(a0)]
            q = policy._queues["action"]
            while len(q) > 1:  # leave exactly one for the next cycle's prev-feed
                actions.append(postprocessor(q.popleft()))
            infer_ms = (time.perf_counter() - t_inf) * 1000.0
            chunk = [a.squeeze(0).cpu().numpy() for a in actions]

            # Skip = ticks executed while this obs aged: sends since the frame
            # was grabbed + the frame's own staleness converted to ticks.
            skip = (executor.sent_count - sent_at_obs) + int(round(staleness_ms * fps / 1000.0))
            k = executor.submit(chunk, skip)

            stats["infer"].append(infer_ms)
            stats["skip"].append(k)
            stats["chunk"].append(len(chunk) - k)
            if log_deltas or log_latency:
                d0 = chunk[min(k, len(chunk) - 1)]
                print(
                    f"cycle {cycle:3d} | tick {executor.sent_count:4d} | "
                    f"infer {infer_ms:5.1f}ms | chunk {len(chunk)} skip {k} | "
                    f"staleness +{staleness_ms:4.0f}ms | underruns {executor.underruns} | "
                    f"Δ0 [{d0[0] * 1000:+5.1f} {d0[1] * 1000:+5.1f} {d0[2] * 1000:+5.1f}]mm "
                    f"grip ({d0[9]:+.3f}, {d0[10] if len(d0) > 10 else 0.0:+.3f})",
                    flush=True,
                )
            if dump_dir is not None:
                cv2.imwrite(str(dump_dir / f"obs_{executor.sent_count:05d}.png"),
                            cv2.cvtColor(img_now, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_PNG_COMPRESSION, 1])
                with open(dump_dir / "state.jsonl", "a") as f:
                    f.write(_json.dumps({"step": int(executor.sent_count),
                                         "state": [float(v) for v in grip_now]}) + "\n")
                # Trajectory telemetry: measured FK vs integrated commanded
                # pose — the "real numbers" for speed/overshoot/tracking
                # analysis (DiffusionPolicy/compare_traj.py).
                arm_state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
                p_cmd, R_cmd = executor.cmd_pose()
                with open(dump_dir / "traj.jsonl", "a") as f:
                    f.write(_json.dumps({
                        "t": time.perf_counter() - episode_start,
                        "tick": int(executor.sent_count),
                        "meas": [float(arm_state.x), float(arm_state.y), float(arm_state.z)],
                        "meas_r6d": [float(v) for v in arm_state.r6d],
                        "cmd": [float(v) for v in p_cmd] if p_cmd is not None else None,
                        "cmd_r6d": [float(v) for v in rotation_matrix_to_rotation_6d_numpy(
                            R_cmd.reshape(1, 3, 3))[0]] if R_cmd is not None else None,
                        "grip_cmd": [float(chunk[-1][9]),
                                     float(chunk[-1][10]) if chunk[-1].shape[0] > 10 else 0.0],
                        "grip_obs": [float(v) for v in grip_now],
                    }) + "\n")

            if cycle % max(success_check_freq // n_act, 1) == 0:
                status = arm_stub.GetSuccessStatus(arm_pb2.SuccessStatusRequest())
                if status.goal_reached:
                    success = True
                    break
            cycle += 1
    finally:
        executor.stop()

    if executor.frozen is not None:
        print(f"\nARM MOTION FROZEN: {executor.frozen}\nReset/re-home before the next episode.",
              flush=True)
    status = arm_stub.GetSuccessStatus(arm_pb2.SuccessStatusRequest())
    success = success or status.goal_reached
    if log_latency and stats["infer"]:
        inf = np.array(stats["infer"])
        sk = np.array(stats["skip"])
        total = executor.sent_count + executor.underruns
        print(
            f"ASYNC SUMMARY | ticks sent {executor.sent_count} at {fps:.0f}Hz | "
            f"underrun ticks {executor.underruns} "
            f"({100.0 * executor.underruns / max(total, 1):.0f}%) | "
            f"replans {len(inf)} (every ~{executor.sent_count / max(len(inf), 1):.1f} ticks) | "
            f"infer p50 {np.percentile(inf, 50):.0f}ms p95 {np.percentile(inf, 95):.0f}ms | "
            f"skip p50 {np.percentile(sk, 50):.0f} | rejected {executor.n_rejected} | "
            f"clamped {executor.n_clamped}",
            flush=True,
        )
    return {
        "success": success,
        "steps": executor.sent_count,
        "displacement_mm": status.cube_displacement * 1000,
        "duration_s": time.perf_counter() - episode_start,
        "n_clamped": executor.n_clamped,
        "n_rejected": executor.n_rejected,
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

    # Persistent camera stream: opened ONCE for the whole run. Opening a fresh
    # gRPC stream per observation blocked ~200 ms/step (measured), silently
    # turning the 50 Hz control loop into a ~5 Hz one.
    camera = CameraStream(gripper_stub, gripper_pb2)
    camera.get()  # block until the first frame arrives
    logger.info("Camera stream up")

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

        # Re-set the gripper opening between episodes. arm_stub.Reset() only
        # re-randomizes the arm/cube; without this, the gripper retains the
        # closed state from the previous episode's grasp + lift. The opening
        # must match the demos' typical FIRST-FRAME state (--start_gripper):
        # a fully-open (0,0) start is itself out of distribution for datasets
        # recorded with a partially-squeezed trigger (real Grabette demos).
        gripper_stub.SendMotorCommand(gripper_pb2.MotorCommand(
            motor1_goal=args.start_gripper[0], motor2_goal=args.start_gripper[1]))

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

        if args.async_exec:
            if joint_mode or use_relative_proprio:
                raise SystemExit("--async_exec supports gripper-only (2D state) cartesian models only.")
            result = run_episode_async(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                arm_stub=arm_stub,
                gripper_stub=gripper_stub,
                arm_pb2=arm_pb2,
                gripper_pb2=gripper_pb2,
                camera=camera,
                device=device,
                max_steps=args.max_steps,
                fps=args.fps,
                success_check_freq=args.success_check_freq,
                task=args.task,
                clamp_pos_m=(args.clamp_pos_mm / 1000.0) if args.clamp_pos_mm else None,
                clamp_rot_rad=(np.deg2rad(args.clamp_rot_deg)) if args.clamp_rot_deg else None,
                log_deltas=args.log_deltas,
                log_latency=args.log_latency,
                dump_dir=(f"{args.dump_obs}/ep{ep:03d}" if args.dump_obs else None),
            )
        else:
            result = run_episode(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            arm_stub=arm_stub,
            gripper_stub=gripper_stub,
            arm_pb2=arm_pb2,
            gripper_pb2=gripper_pb2,
            camera=camera,
            device=device,
            max_steps=args.max_steps,
            fps=args.fps,
            success_check_freq=args.success_check_freq,
            debug=args.debug,
            log_gripper=args.log_gripper,
            log_deltas=args.log_deltas,
            log_latency=args.log_latency,
            dump_dir=(f"{args.dump_obs}/ep{ep:03d}" if args.dump_obs else None),
            use_relative_proprio=use_relative_proprio,
            start_pos=start_pos,
            start_rot=start_rot,
            task=args.task,
            joint_mode=joint_mode,
            clamp_pos_m=(args.clamp_pos_mm / 1000.0) if args.clamp_pos_mm else None,
            clamp_rot_rad=(np.deg2rad(args.clamp_rot_deg)) if args.clamp_rot_deg else None,
            max_ticks=args.max_ticks,
            skip_stale=args.skip_stale,
        )
        # On the REAL arm GetSuccessStatus is a stub (no object tracking), so
        # result["success"] is meaningless there: ask the operator instead and
        # append every episode to a JSONL so A/B sessions produce real numbers.
        if args.ask_success:
            ans = input(f"  Episode {ep + 1}: grasp success? [y/N] ").strip().lower()
            result["success"] = ans in ("y", "yes", "o", "oui")
            with open(args.ask_success, "a") as f:
                f.write(_json.dumps({
                    "episode": ep, "success": result["success"],
                    "steps": result["steps"], "checkpoint": args.checkpoint,
                    "n_action_steps": args.n_action_steps, "fps": args.fps,
                }) + "\n")
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
    camera.stop()
    arm_channel.close()
    gripper_channel.close()


if __name__ == "__main__":
    main()
