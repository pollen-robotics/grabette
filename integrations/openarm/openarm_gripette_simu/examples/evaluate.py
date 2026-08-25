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
import atexit
import logging
import math
import threading
import time

import cv2
import grpc
import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
import json as _json
from pathlib import Path as _Path

from lerobot.policies import get_policy_class


def _load_policy_any(checkpoint: str):
    """Load any LeRobot policy from a checkpoint (local dir or Hub repo id),
    dispatching on the `type` field in its config — the same eval works for
    Diffusion / ACT / Pi0.5 / Pi0Fast.

    VLA-specific handling (mirrors the smoke scripts and the Ficelle server):
    compile is forced OFF (a train-time optimization; on the pi05 port it
    also triggers an inductor dtype crash), and pi05/pi0 weights are cast to
    float32 — their checkpoints are saved bf16, and the pi05 port's flow
    path has a bf16 dtype clash at inference.
    """
    from lerobot.configs import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if hasattr(cfg, "compile_model"):
        cfg.compile_model = False
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
    if cfg.type in ("pi05", "pi0"):
        policy = policy.to(dtype=torch.float32)
    return policy


def _policy_img_hw(policy):
    """(H, W) the checkpoint was trained on. ACT has no internal resize
    (diffusion/pi0 do), so live frames must be brought to this size."""
    for ft in policy.config.image_features.values():
        return int(ft.shape[1]), int(ft.shape[2])
    return None
from scipy.spatial.transform import Rotation
from openarm_gripette_simu.rotation import (
    rotation_6d_to_matrix as rotation_6d_to_rotation_matrix_numpy,
    rotation_matrix_to_6d as rotation_matrix_to_rotation_6d_numpy,
)
from gripette.grasp_projection import GraspProjection, clamp_to_command_limits


# --- Debug frame display (headless-safe) -------------------------------------
# cv2.imshow needs a GUI-enabled OpenCV. The workspace often resolves to
# opencv-python-headless (pulled by lerobot; same cv2 namespace), whose imshow
# raises cv2.error. So --debug shows a live window when a GUI build is present,
# and otherwise falls back to writing numbered frames to disk — either way it
# never crashes the eval.
_DEBUG_GUI = None       # None = untried, True = window works, False = headless
_DEBUG_N = 0
_DEBUG_DIR = _Path("eval_debug_frames")


def debug_display(img_bgr):
    """Show one eval frame in a window, or save it if OpenCV has no GUI. Warns once."""
    global _DEBUG_GUI, _DEBUG_N
    if _DEBUG_GUI is not False:
        try:
            cv2.imshow("Evaluation", img_bgr)
            cv2.waitKey(1)
            _DEBUG_GUI = True
            return
        except cv2.error:
            _DEBUG_GUI = False
            _DEBUG_DIR.mkdir(exist_ok=True)
            print(f"[--debug] OpenCV has no GUI (headless build); saving frames "
                  f"to {_DEBUG_DIR}/ instead of a live window.", flush=True)
    cv2.imwrite(str(_DEBUG_DIR / f"frame_{_DEBUG_N:05d}.png"), img_bgr)
    _DEBUG_N += 1


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


def apply_grip_gain(g1, g2, gain, ref):
    """Gripper actuator calibration (--grip_gain): scale closure depth around
    the open reference so open stays open and closes deepen.

    The recorded close values are positions of the Grabette's trigger-driven
    linkage squeezing deformable fingertips; the deployed Feetech servo
    chasing the same position numbers delivers less squeeze (position control,
    bounded torque, no force feedback). This maps model-units to
    servo-effective-units at send time.

    At gain 1.0 this MUST be a pure identity: gripper sign/range conventions
    vary by model (the sim models close with NEGATIVE proximal values, real
    models with positive ones) — an unconditional clamp here silently broke
    every sim close to "open" (regression found 2026-07-12 via the sim
    re-eval of diffusion_grabette_simu_release). The clamp only applies to
    gain-scaled values, symmetric so it is convention-agnostic.
    """
    if gain == 1.0:
        return float(g1), float(g2)
    g1 = ref[0] + gain * (g1 - ref[0])
    g2 = ref[1] + gain * (g2 - ref[1])
    return float(np.clip(g1, -1.6, 1.6)), float(np.clip(g2, -1.6, 1.6))


# Gripper joint limits in ROBOT FRAME, mirroring the AUTHORITY —
# gripette.config.Settings.motor{1,2}_max. Use the same radians expression, not
# the rounded values the docs quote (1.484 / 2.025): those are LARGER than the
# real limits, so commands clamped to them are rejected by the service, and the
# error message hides it by printing the limit rounded ("Motor 1 goal 1.484 rad
# outside limits [0.000, 1.484]" — seen on the real gripper 2026-07-28).
# A device may narrow these via GRIPPER_MOTOR*_MIN/MAX; the service remains the
# authority and rejects anything out of range, so this is only a sanity bound.
GRIPPER_LIMITS = ((0.0, math.radians(85)), (0.0, math.radians(116)))


class GripAssist:
    """Load-triggered grip assist for the UNDER-CLOSE failure.

    The policy learned the DEMONSTRATED closing angles — which is where the
    human's fingers sat while PRESSING the object. Replaying that angle on a
    force-blind position servo can leave the fingers stopped just short,
    touching nothing: "parked at angle θ in air" and "pressing the object at
    angle θ" are indistinguishable to the policy, and the dataset gave it no
    force channel to tell them apart (measured on the bench 2026-07-27).

    So leave the policy fully in charge of WHEN to close and WHICH posture to
    use — with a 2-DoF index against a fixed thumb the proximal/distal ratio
    IS the grasp type (fingertip pinch ↔ power wrap), and a fixed target pose
    would both override that choice and risk driving the index into the thumb.
    Intervene ONLY in the one state where the policy demonstrably fails: it
    has SETTLED into a closed pose that is NOT gripping. Then extend the
    closure ALONG THE POLICY'S OWN POSTURE DIRECTION until the load reports
    contact — object-adaptive, so there is no per-object gain to tune —
    bounded by a max extra travel, the joint limits, and the servo torque cap.

    Why load discriminates (bench-verified): while commanding a MODERATE goal
    the empty gripper can reach, a free finger arrives and its load falls to
    ~0, while a finger blocked by an object keeps a position error and its
    load stays pinned near the cap. (Slamming an unreachable goal destroys the
    signal: then BOTH cases stall at the cap. Hence "extend gradually", never
    "drive to max".)

    CRITICAL: load ALONE is a weak test, because there is no torque sensor —
    "load" is a PWM/current proxy, so a MOVING finger reads a nonzero motion
    load whether or not it touches anything (measured while stepping: up to
    ~88; at rest with the jaws free: ~0-24).

    So contact is detected as a STALL — commanded to advance, but not
    advancing, while drawing effort:

        lag = |commanded - measured|  >  assist_lag  AND  load >= threshold

    Bench numbers backing that (rigid object, 25% cap, 0.02 rad steps): free
    motion lag 0.002-0.004 rad with load 0-24; from first contact lag jumps to
    0.014 and GROWS (0.022, 0.031, 0.041, 0.052) while load climbs 72 -> 250.
    Lag is used rather than a raw velocity/ratio because it is rate-robust: a
    blocked finger's lag grows by the commanded increment every tick no matter
    how much time the servo had, whereas an absolute velocity threshold shifts
    with the control-loop rate. The load term keeps the degenerate case honest:
    a limp/disabled servo is also "not moving", but draws nothing, and must NOT
    read as a grasp.

    Readings are debounced (assist_confirm_ticks): one noisy tick must not
    latch GRIPPED, which would leave exactly the under-grip this fixes.

    Detecting contact is not the same as HOLDING: the stall fires at light
    touch (measured on the bench: lag 0.0114, load 64, both barely over
    threshold), and grip force only persists while commanding PAST the object.
    So on contact we press a further `squeeze` before latching — pressing
    0.04-0.08 rad past contact took load 72 -> 250 (the cap) on the bench, and
    the torque cap bounds the force, so this is where a light touch becomes a
    grip that survives a lift.

    States: IDLE → ASSISTING → SQUEEZING → GRIPPED (offset latched; re-assists
    if the load later drops = slip) or EXHAUSTED (no contact within max_extra →
    assist released; re-arms only once the policy opens again, so it cannot
    chatter). A policy command back toward open always releases the assist —
    a deliberate release is honored.
    """

    def __init__(self, load_thresh, ref, min_close=0.15, stable_ticks=5,
                 stable_eps=0.01, step=0.02, max_extra=0.4,
                 dwell_ticks=2, confirm_ticks=2, lag=0.010, settle_eps=0.004,
                 squeeze=0.05, limits=GRIPPER_LIMITS):
        # --- tunables -----------------------------------------------------
        self._thresh = float(load_thresh)          # load floor for "pushing"
        self._ref = np.asarray(ref, dtype=float)   # open pose (--start_gripper)
        self._min_close = float(min_close)         # displacement counting as closing
        self._stable_ticks = int(stable_ticks)     # ticks the CMD must hold steady
        self._stable_eps = float(stable_eps)       # per-tick cmd change = steady
        self._step = float(step)                   # extra closure per increment
        self._max_extra = float(max_extra)         # cap on total extra closure
        self._dwell_ticks = int(dwell_ticks)       # settle ticks after an increment
        self._confirm_ticks = int(confirm_ticks)   # debounce on contact / slip
        self._lag = float(lag)                     # cmd-vs-measured = blocked
        self._settle_eps = float(settle_eps)       # per-tick motion = settled
        self._squeeze = float(squeeze)             # press past contact before latching
        self._limits = limits
        # --- state --------------------------------------------------------
        self.state = "IDLE"
        self.offset = 0.0          # extra closure along the policy's posture
        self.trigger_step = None   # step at which the assist last engaged
        self.n_gripped = 0         # contacts found (telemetry)
        self.n_exhausted = 0       # top-ups that found nothing (telemetry)
        self.lag = self.advance = float("nan")   # last measurements (logging)
        self.why = "init"          # reason for the current state (logging)
        self._squeeze_to = 0.0     # offset to reach while SQUEEZING
        self._dwell = 0
        self._confirm = 0
        self._stable = 0
        self._prev = None          # previous policy command (steadiness test)
        self._meas = None          # previous measured position (settle test)
        self._sent = None          # last pose we actually commanded
        self._armed = True         # cleared after EXHAUSTED until re-open

    def update(self, cmd, load, step=None, meas=None):
        """One tick. cmd = the policy's (prox, distal) goal, load = the
        gripper's (prox, distal) present_load, meas = its measured (prox,
        distal) position (for the stall test; if omitted, falls back to a
        load-only test). Returns the (prox, distal) to actually SEND."""
        cmd = np.asarray(cmd, dtype=float)
        peak = max(abs(float(x)) for x in load) if load is not None else 0.0
        # STALL = the fingers have STOPPED MOVING, short of the pose we
        # commanded, while the motor draws effort. All three terms are needed:
        #   - SETTLED: a finger still travelling to a freshly-commanded pose
        #     legitimately lags a lot and draws cap-level current. Without this
        #     term the transit to the policy's own close reads as contact
        #     (observed on the bench: lag 0.09, load 250, 3 ticks in, zero
        #     top-up). Command-stable is NOT the same as fingers-settled.
        #   - LAG: settled short of the commanded pose = something is in the way
        #     (a finger that simply arrived has ~no lag).
        #   - LOAD: rejects a limp/disabled servo, which is also settled+lagging
        #     but pushing nothing.
        meas_arr = None if meas is None else np.asarray(meas, dtype=float)
        if meas_arr is not None and self._sent is not None and self._meas is not None:
            lag = float(np.max(np.abs(np.asarray(self._sent, dtype=float) - meas_arr)))
            advance = float(np.max(np.abs(meas_arr - self._meas)))
            self.lag, self.advance = lag, advance
            advance_str = f"{advance:.4f}"
            stalled = (advance < self._settle_eps
                       and lag > self._lag
                       and peak >= self._thresh)
        else:
            self.lag = self.advance = float("nan")
            advance_str = "n/a"
            stalled = False if meas_arr is not None else peak >= self._thresh
        if meas_arr is not None:
            self._meas = meas_arr.copy()
        d = cmd - self._ref
        # CONVENTION-AGNOSTIC: "closing" = displaced from the open reference by
        # this much in ANY direction, measured as a magnitude. Current models
        # close POSITIVE, but legacy pre-flip datasets/models close NEGATIVE
        # proximal (the PROXIMAL_CMD_SIGN=-1 server bridge covers the sim side
        # only — it never reaches this client-side logic). Signed tests here
        # would silently never fire for those, and hardcoded sign conventions
        # in shared paths are this project's most recurrent bug class.
        closing = float(np.linalg.norm(d)) >= self._min_close

        # Track "the policy's close has settled" on the COMMAND (its intent),
        # not the position — a blocked finger's position settles even while the
        # policy is still driving deeper.
        if self._prev is not None and float(np.max(np.abs(cmd - self._prev))) < self._stable_eps:
            self._stable += 1
        else:
            self._stable = 0
        self._prev = cmd.copy()

        if not closing:
            # Policy opened (or never closed): release everything, re-arm.
            self.state, self.offset, self._armed = "IDLE", 0.0, True
            self._confirm = self._dwell = 0
            self.why = f"open (|cmd-ref| {float(np.linalg.norm(d)):.3f} < {self._min_close})"
            self._sent = self._clamp(cmd)
            return self._sent

        if self.state == "IDLE":
            if not self._armed:
                self.why = "disarmed (waiting for the policy to open)"
            elif self._stable < self._stable_ticks:
                self.why = f"cmd still moving ({self._stable}/{self._stable_ticks} steady)"
            if self._armed and self._stable >= self._stable_ticks:
                if stalled:
                    self.state = "GRIPPED"      # already gripping — hands off
                    self.n_gripped += 1
                    self.why = "already gripping at the policy's own command"
                else:
                    self.state = "ASSISTING"    # settled but empty — top up
                    self.trigger_step = step
                    self._dwell = self._confirm = 0
                    self.why = "settled closed but not gripping — topping up"
        elif self.state == "ASSISTING":
            # Brief dwell after each increment so the fingers have a chance to
            # move before we judge whether they are stuck (kept small: the stall
            # test already tolerates motion, unlike a bare load threshold).
            if self._dwell < self._dwell_ticks:
                self._dwell += 1
                self.why = f"dwell {self._dwell}/{self._dwell_ticks} after a step"
            elif stalled:
                self._confirm += 1
                if self._confirm >= self._confirm_ticks:
                    # Contact found — but that is only a TOUCH. Press on for
                    # `squeeze` more to develop actual grip force before
                    # latching (bounded by the torque cap and max_extra).
                    self._squeeze_to = min(self.offset + self._squeeze,
                                           self._max_extra)
                    self.why = f"contact confirmed (lag {self.lag:.4f}, load {peak:.0f})"
                    self.state = ("SQUEEZING" if self._squeeze_to > self.offset
                                  else "GRIPPED")
                    if self.state == "GRIPPED":
                        self.n_gripped += 1
                    self._confirm = 0
            elif self.offset + self._step <= self._max_extra:
                self._confirm = 0
                self.offset += self._step
                self._dwell = 0
                self.why = (f"stepping to {self.offset:.3f} (settled={advance_str}, "
                            f"lag {self.lag:.4f}, load {peak:.0f})"
                            if self.lag == self.lag else f"stepping to {self.offset:.3f}")
            else:
                # Nothing within reach: release and wait for the policy to
                # re-approach (it must open first, so this cannot chatter).
                self.state, self.offset, self._armed = "EXHAUSTED", 0.0, False
                self.n_exhausted += 1
                self.why = f"no contact within max_extra {self._max_extra:.3f} — released"
        elif self.state == "SQUEEZING":
            # Press past contact to build grip force, then latch. No load check
            # here: we are deliberately pushing into a known object, and the
            # torque cap is what limits the force.
            if self.offset + self._step <= self._squeeze_to:
                self.offset += self._step
                self.why = f"squeezing to {self._squeeze_to:.3f} (at {self.offset:.3f})"
            else:
                self.state = "GRIPPED"
                self.n_gripped += 1
                self._confirm = 0
                self.why = f"latched at {self.offset:.3f} (load {peak:.0f})"
        elif self.state == "GRIPPED":
            # Slip = the fingers stopped PUSHING (load fell). Deliberately does
            # NOT reuse the `stalled` test: that requires the fingers to be
            # settled, but a gripped finger legitimately keeps creeping while
            # the policy raises its own command or the object beds in. Treating
            # that motion as slip made the assist re-close on a grasp already
            # pushing at the cap, ratcheting the offset 0.18 -> 0.24 -> 0.30 on
            # the real arm (2026-07-28) and tripping the arm's lead guard.
            # Debounced so one noisy reading cannot restart a top-up.
            if peak < self._thresh:
                self._confirm += 1
                self.why = f"grip lost? {self._confirm}/{self._confirm_ticks} (load {peak:.0f})"
                if self._confirm >= self._confirm_ticks:
                    self.state, self._dwell, self._confirm = "ASSISTING", 0, 0
                    self.why = "slip — re-closing"
            else:
                self._confirm = 0
                self.why = f"holding (load {peak:.0f}, lag {self.lag:.4f})"
        elif self.state == "EXHAUSTED":
            # Hold at the policy's own command until it opens (kept explicit so
            # the diagnostic never goes stale while disarmed).
            self.why = "disarmed after finding nothing — waiting for the policy to open"

        out = self._clamp(cmd + self._direction(d) * self.offset)
        self._sent = out            # lag next tick is measured against this
        return out

    def _direction(self, d):
        """Unit vector along the policy's commanded closing posture, so extra
        travel deepens the grasp it chose instead of imposing a pose."""
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-6 else np.zeros_like(d)

    # Stay a hair INSIDE the limit even so: the goal crosses the wire as
    # float32, and a value exactly at the boundary can round up over it.
    _LIMIT_MARGIN = 1e-4

    def _clamp(self, v):
        """Bound travel by MAGNITUDE, preserving sign — so this works for both
        the current positive-closing convention and legacy negative-closing
        models. (The gripette service is the real authority on limits and
        rejects out-of-range goals itself; this only stops the assist from
        driving past the mechanical range.)"""
        return tuple(float(np.clip(v[i],
                                   -(self._limits[i][1] - self._LIMIT_MARGIN),
                                   self._limits[i][1] - self._LIMIT_MARGIN))
                     for i in (0, 1))


logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Gripette policy on simulator")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    p.add_argument("--policy_addr", type=str, default=None,
                   help="HOST:PORT of a ficelle policy server (websocket transport) OR "
                        "an iroh ticket (zero-config remote, printed by `serve.py "
                        "--transport iroh`); replaces local policy inference at chunk "
                        "granularity (sync cartesian mode only)")
    p.add_argument("--jpeg_quality", type=int, default=None, help="encode images as JPEG at this quality (1-100) before sending")
    p.add_argument("--resize", action="store_true", default=False, help="downscale images to the model's resize resolution before sending (transport-only; requires a server whose policy resizes internally, e.g. diffusion)")
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
    p.add_argument("--latch_close", type=float, default=None, metavar="THRESH",
                   help="Async mode: once any gripper command exceeds THRESH, hold the "
                        "running max (the close becomes irreversible). Legitimate for "
                        "end-at-lift datasets where no demo reopens near the object: "
                        "diffusion re-draws its mode at every replan, so a multi-chunk close "
                        "gets undone by the next draw — the latch turns one sampled "
                        "close into a completed one. Suggested: 0.55.")
    p.add_argument("--commit_close", type=float, default=None, metavar="THRESH",
                   help="Async mode: when a drawn chunk's gripper command crosses THRESH "
                        "anywhere in the chunk, execute that chunk TO THE END without "
                        "replanning. The model initiates a close almost only by continuing "
                        "one it observes (measured P(initiate|static pre-grasp) ~ 0-15%% vs "
                        "100%% once the state shows a started close); replanning every few "
                        "ticks executes only chunk heads and re-rolls the decision, so the "
                        "close stays forever in the receding tail. Committing lets the "
                        "fingers actually move; the state feedback then carries the close. "
                        "Complementary to --latch_close. Suggested: 0.55.")
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
    p.add_argument("--no_reset", action="store_true",
                   help="Do NOT move the arm at episode start (skip the server Reset). "
                        "Workflow: torque off, place the arm by hand, torque on "
                        "(set_arm_torque.py — the server resyncs its integrator on "
                        "torque-on), then run with this flag: the episode starts from "
                        "wherever the arm is.")
    p.add_argument("--home_joints", type=float, nargs=7, default=None,
                   metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
                   help="Episode start joint configuration (rad), passed to the arm "
                        "server's Reset. CRITICAL for camera-local-delta policies: the "
                        "deltas are relative, so the start pose ANCHORS the whole "
                        "trajectory — the start CAMERA VIEW must match the demos' "
                        "first frames (position AND pitch). Find it by jogging the arm "
                        "until the live view matches a demo start frame, then reading "
                        "the joints (examples/read_arm_state.py). Default: the server's "
                        "built-in home.")
    p.add_argument("--grip_gain", type=float, default=1.0,
                   help="Actuator calibration for the gripper: scale the model's gripper "
                        "commands around the open reference (--start_gripper), so open "
                        "stays open and CLOSES DEEPEN. The recorded closes are positions "
                        "of the Grabette's trigger linkage squeezing soft fingertips; the "
                        "deployed Feetech servo chasing the same position numbers delivers "
                        "less squeeze (no force feedback). Try 1.3-1.6 if grasps slip. "
                        "Applied at send time, AFTER commit/latch logic (those thresholds "
                        "stay in model units).")
    p.add_argument("--grip_torque_limit", type=float, default=0.0,
                   help="Per-grasp GRIP FORCE CAP as a fraction 0..1 of the servo's max "
                        "torque, applied to BOTH gripper DOFs. 0 (default) = unset = full "
                        "torque = exactly today's behavior. The policy's per-DOF position "
                        "targets are unchanged (they encode grasp SHAPE); the DOF driven "
                        "into the object stalls at this cap, giving an object-size-"
                        "independent, consistent grip force without a shape classifier. "
                        "Force is now the cap, not Kp x position-overshoot — so --grip_gain "
                        "only needs to push the closing target past contact. Real hardware "
                        "only (no-op in sim). Try 0.2-0.4; watch motor{1,2}_load telemetry.")
    p.add_argument("--grip_assist", type=float, default=None, metavar="LOAD_THRESH",
                   help="Load-triggered GRIP ASSIST (sync cartesian mode; the recommended "
                        "fix for the under-close failure — supersedes --latch_close/"
                        "--commit_close/--grip_gain when set; pair with --grip_torque_limit). "
                        "The policy replays the demonstrated closing ANGLE, which is where "
                        "the human's fingers sat while PRESSING the object — on a force-blind "
                        "position servo that can stop just short, touching nothing. This "
                        "watches for the policy SETTLING into a closed pose with LOW load "
                        "(= not gripping) and only then extends the closure ALONG THE "
                        "POLICY'S OWN posture until the load reports contact, so the grasp "
                        "TYPE (pinch vs power wrap) stays the policy's choice. LOAD_THRESH "
                        "is the present_load above which we consider the fingers loaded — "
                        "hardware units (STS3215 ~0-1000, so a few hundred) do NOT match "
                        "sim units (MuJoCo actuator_force): measure both, never reuse a "
                        "threshold across them.")
    p.add_argument("--assist_min_close", type=float, default=0.15,
                   help="--grip_assist: the policy command must exceed --start_gripper by "
                        "this much (rad, either DOF) to count as 'closing' — below it the "
                        "assist releases and re-arms (a deliberate open is always honored).")
    p.add_argument("--assist_stable_ticks", type=int, default=5,
                   help="--grip_assist: consecutive ticks the policy's gripper command must "
                        "hold steady before the assist may engage. Guards against firing "
                        "mid-approach; raise it if you see it trigger while still moving.")
    p.add_argument("--assist_stable_eps", type=float, default=0.01,
                   help="--grip_assist: per-tick command change (rad) below which the "
                        "command counts as steady.")
    p.add_argument("--assist_step", type=float, default=0.02,
                   help="--grip_assist: extra closure added per tick (rad) while topping up. "
                        "Smaller = gentler approach to contact.")
    p.add_argument("--assist_lag", type=float, default=0.010,
                   help="--grip_assist: STALL threshold (rad). Contact = the fingers lag the\n                        commanded pose by more than this WHILE drawing load (>= LOAD_THRESH).\n                        Measured on the real gripper at 25%% cap: free motion lags\n                        0.002-0.004 rad, first contact 0.014 and growing — so ~0.010 sits\n                        mid-gap. Rate-robust (a blocked finger's lag grows every tick),\n                        unlike an absolute velocity threshold.")
    p.add_argument("--assist_log", type=str, default=None, metavar="FILE",
                   help="--grip_assist: append a per-tick JSONL trace (model vs sent gripper "
                        "command, observed position, load, assist state/offset/lag/advance and "
                        "the reason for it) — the record needed to debug a grasp attempt "
                        "offline. One file per session; episodes are delimited by the step "
                        "counter restarting.")
    p.add_argument("--assist_squeeze", type=float, default=0.05,
                   help="--grip_assist: extra closure (rad) to press PAST contact before "
                        "latching. Contact fires at a light touch, and grip force only "
                        "persists while commanding past the object — measured on the bench, "
                        "pressing 0.04-0.08 past contact took load 72 -> 250 (cap). The "
                        "torque cap bounds the force. 0 = latch at first touch.")
    p.add_argument("--assist_settle_eps", type=float, default=0.004,
                   help="--grip_assist: per-tick measured movement (rad) below which the "
                        "fingers count as SETTLED. Contact is only judged once settled — a "
                        "finger still travelling to a new commanded pose lags a lot and "
                        "draws cap current, which would otherwise read as contact.")
    p.add_argument("--assist_dwell_ticks", type=int, default=2,
                   help="--grip_assist: ticks to WAIT after each increment before trusting "
                        "the load reading. These servos have no torque sensor — a MOVING "
                        "finger reads a nonzero motion load whether or not it touches "
                        "anything, so the reading is only meaningful once settled.")
    p.add_argument("--assist_confirm_ticks", type=int, default=2,
                   help="--grip_assist: consecutive settled readings needed to accept "
                        "contact (and, once gripped, to accept a slip). Debounces the noisy "
                        "load register — one spurious tick would otherwise latch a grasp "
                        "that isn't there.")
    p.add_argument("--assist_max_extra", type=float, default=0.4,
                   help="--grip_assist: max total extra closure (rad) beyond the policy's "
                        "command. Reaching it without contact = nothing in the jaws → the "
                        "assist releases until the policy re-approaches.")
    p.add_argument("--grasp_projection", choices=["auto", "on", "off"], default="auto",
                   help="Interpret the policy's last two action channels as "
                        "(strategy, closure) from the grasp projection rather than "
                        "raw joint angles, decoding them to angles before sending. "
                        "'auto' inspects the checkpoint's saved normaliser ranges "
                        "(projected channels are 0..1, raw angles reach ~1.6 rad) "
                        "and logs what it decided. Override with on/off if a raw "
                        "dataset happens to stay under 1 rad on both joints.")
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
    args = p.parse_args()
    if args.checkpoint is None and args.policy_addr is None:
        p.error("either --checkpoint or --policy_addr is required")
    if args.grip_assist is not None and args.async_exec:
        p.error("--grip_assist is sync-mode only (not --async_exec)")
    return args


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

    # A live camera delivers 15-30 Hz with hiccups <0.5 s; a frame older than
    # this is a DEAD/FROZEN stream. A policy fed a frozen frame runs open-loop
    # while the arm keeps integrating deltas ("crazy arm", 2026-07-16) — that
    # must be a hard stop, not a log line.
    STALE_LIMIT_S = 2.0

    def __init__(self, gripper_stub, gripper_pb2):
        self._stub = gripper_stub
        self._pb2 = gripper_pb2
        self._lock = threading.Lock()
        self._latest = None
        # Decoded present_load telemetry, kept separate from the policy state
        # tuple (must NOT enter the observation — it would change obs dim).
        self._latest_load = (0.0, 0.0)
        self._last_frame_wall = None  # local monotonic time of the last frame
        self._bad_frames = 0  # undecodable-payload counter (rate-limits the log)
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
                    img_bgr = (cv2.imdecode(
                        np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                        if len(frame.jpeg_data) else None)
                    if img_bgr is None:
                        # Service up but camera not producing images (empty or
                        # corrupt JPEG payload). Do NOT crash-reconnect-churn:
                        # keep reading motor state messages; the staleness
                        # watchdog in get()/get_pair() turns this into a hard,
                        # explained stop on the consumer side.
                        self._bad_frames += 1
                        if self._bad_frames % 30 == 1:  # ~1 log/s at 30 Hz
                            logger.error(
                                "Camera frame has no decodable image (empty/"
                                "corrupt JPEG) — is the Gripette camera actually "
                                "streaming? (GRIPPER_CAMERA_MODE, service logs)")
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    gripper = np.array(
                        [frame.motor_state.motor1_position,
                         frame.motor_state.motor2_position],
                        dtype=np.float32,
                    )
                    with self._lock:
                        self._latest = (img_rgb, gripper, float(frame.timestamp_ms))
                        self._latest_load = (float(frame.motor_state.motor1_load),
                                             float(frame.motor_state.motor2_load))
                        self._last_frame_wall = time.monotonic()
                        self._history.append(self._latest)
                        if len(self._history) > 8:
                            self._history.pop(0)
                    self._ready.set()
            except Exception as e:  # noqa: BLE001 — stream drop: reconnect
                if not self._stop:
                    logger.warning(f"Camera stream dropped ({e}); reconnecting...")
                    time.sleep(0.2)

    def _check_fresh(self, timeout: float):
        """Hard safety gate for every frame handed to the policy: raise with an
        actionable message if the camera never streamed, or streamed once and
        FROZE (the classic failure: gripper service up, camera module not
        started — the old code silently served the last frame forever and the
        policy ran open-loop)."""
        if not self._ready.wait(timeout):
            raise RuntimeError(
                f"No frame from the Gripette camera within {timeout:.0f}s — the "
                "service answered but the camera is not streaming. Check the "
                "gripette service: GRIPPER_CAMERA_MODE=video, camera cable, "
                "service logs (journalctl -u gripette).")
        with self._lock:
            age = time.monotonic() - self._last_frame_wall
        if age > self.STALE_LIMIT_S:
            raise RuntimeError(
                f"Camera stream FROZEN: newest frame is {age:.1f}s old "
                f"(limit {self.STALE_LIMIT_S:.0f}s). Refusing to feed a stale "
                "image to the policy — the arm would run open-loop. Check the "
                "gripette camera service, then restart the episode.")

    def get(self, timeout: float = 5.0):
        self._check_fresh(timeout)
        with self._lock:
            return self._latest

    def get_load(self):
        """Latest decoded gripper present_load (motor1, motor2). Telemetry
        only — 0 in sim, real effort on hardware."""
        with self._lock:
            return self._latest_load

    def get_pair(self, timeout: float = 5.0):
        """Return (previous_frame, latest_frame) — the two most recent DISTINCT
        camera frames, each (img_rgb, gripper, ts_ms). The closest available
        approximation of the 20 ms-spaced observation pair the policy was
        trained on (camera period sets the floor). Duplicates latest if only
        one frame exists yet."""
        self._check_fresh(timeout)
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
                 start_pos=None, start_rot=None, latch_close=None,
                 grip_gain=1.0, grip_ref=(0.0, 0.0), grip_torque_limit=0.0):
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
        # Close-latch: in end-at-lift datasets, closing is IRREVERSIBLE by
        # construction (no demo reopens near the object) — but diffusion
        # re-draws its mode every replan, so a close that needs several
        # consecutive chunks gets undone by the next draw (measured: distal
        # 0.86 for one chunk, then reopened). Once any gripper command
        # exceeds the threshold, hold the running max — one sampled close
        # becomes a completed close.
        self._latch_close = latch_close
        self._grip_latch = None
        self.latched_at_tick = None
        self._grip_gain = grip_gain
        self._grip_ref = grip_ref
        self._grip_torque_limit = grip_torque_limit
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

    def remaining(self):
        """Actions of the current chunk not yet sent (thread-safe)."""
        with self._lock:
            return max(len(self._chunk) - self._i, 0)

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
                    g1 = float(grip[0])
                    g2 = float(grip[1]) if len(grip) > 1 else 0.0
                    if self._latch_close is not None:
                        if self._grip_latch is None and max(g1, g2) > self._latch_close:
                            self._grip_latch = [g1, g2]
                            self.latched_at_tick = self.sent_count
                        if self._grip_latch is not None:
                            self._grip_latch = [max(self._grip_latch[0], g1),
                                                max(self._grip_latch[1], g2)]
                            g1, g2 = self._grip_latch
                    # Gain AFTER latch: latch/commit thresholds are in model
                    # units; the gain is actuator calibration at send time.
                    g1, g2 = apply_grip_gain(g1, g2, self._grip_gain, self._grip_ref)
                    # Gripper: fire-and-forget future — a blocking round trip
                    # to the Pi over WiFi would eat the 20 ms tick budget.
                    self._gripper_stub.SendMotorCommand.future(
                        self._gripper_pb2.MotorCommand(
                            motor1_goal=g1, motor2_goal=g2,
                            motor1_torque_limit=self._grip_torque_limit,
                            motor2_torque_limit=self._grip_torque_limit,
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


def detect_grasp_projection(checkpoint):
    """Was this checkpoint trained on (strategy, closure) instead of raw angles?

    Returns True / False, or None when it cannot be determined.

    Detected from the SAVED NORMALISER RANGES, not from channel names: the
    checkpoint records only feature shapes, so the names are gone by this point
    (`output_features = {"action": {"type": "ACTION", "shape": [11]}}`).

    The discriminator is that both projected channels are normalised to [0, 1]
    whereas raw angles are radians reaching ~1.6. Measured: the projected
    checkpoint has action.max[-2:] = (0.813, 1.000); the raw mustard dataset has
    (1.338, 1.448).

    KNOWN FAILURE MODE, hence the override flag: a raw dataset in which NEITHER
    gripper joint ever exceeded 1 rad (57 deg) in any frame would be misread as
    projected. The decision is logged loudly so it can be caught at a glance.
    """
    if not checkpoint:
        # Remote inference (--policy_addr with no --checkpoint): there is nothing
        # local OR on the Hub to inspect, so auto-detection is impossible. The
        # caller turns None into a hard error demanding --grasp_projection.
        return None
    stats_path = None
    ckpt = _Path(checkpoint)
    if ckpt.is_dir():
        files = sorted(ckpt.glob("*normalizer*.safetensors"))
        stats_path = str(files[0]) if files else None
    else:
        # A Hub id. Fetch JUST the normaliser file (a few KB) rather than giving
        # up: returning None here used to mean "assume raw angles", which sends a
        # closure of 1.0 as 1.0 RADIAN and silently reproduces the under-close —
        # and evaluating a Hub checkpoint is the normal case after a cloud run.
        try:
            from huggingface_hub import list_repo_files, hf_hub_download

            names = [f for f in list_repo_files(checkpoint)
                     if "normalizer" in f and f.endswith(".safetensors")]
            if names:
                stats_path = hf_hub_download(checkpoint, sorted(names)[0])
        except Exception as e:
            logger.warning(f"Could not fetch normaliser stats from the Hub ({e})")
    if stats_path is None:
        return None
    try:
        from safetensors.torch import load_file

        stats = load_file(stats_path)
    except Exception as e:
        logger.warning(f"Could not read normaliser stats ({e}); "
                       "pass --grasp_projection on|off explicitly")
        return None
    amax = stats.get("action.max")
    if amax is None or len(amax) < 2:
        return None
    last_two = [float(v) for v in amax.flatten()[-2:]]
    projected = all(v <= 1.0 + 1e-3 for v in last_two)
    logger.info(f"Grasp projection auto-detect: action.max[-2:] = "
                f"{[round(v, 4) for v in last_two]} -> "
                f"{'PROJECTED (strategy, closure)' if projected else 'RAW angles'}")
    return projected


def build_observation(
    arm_stub,
    arm_pb2,
    camera,
    use_relative_proprio,
    start_pos,
    start_rot,
    joint_mode=False,
    grasp_projection=None,
):
    """Build the full observation (camera image + state) for one step.

    Returns (camera_image, state, frame_ts_ms)."""
    camera_image, gripper_joints, frame_ts_ms = camera.get()

    if grasp_projection is not None:
        # The policy was TRAINED on (strategy, closure) proprioception, so the
        # live gripper position has to be encoded the same way. Feeding raw
        # angles here would silently put the state channel out of distribution —
        # no error, just a worse policy.
        _s, _c = grasp_projection.encode(float(gripper_joints[0]),
                                         float(gripper_joints[1]))
        gripper_joints = np.array(
            [0.0 if math.isnan(_s) else _s, _c], dtype=np.float32)

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
    grip_gain=1.0,
    grip_ref=(0.0, 0.0),
    grip_torque_limit=0.0,
    latch_close=None,
    grip_assist=None,
    assist_log=None,
    grasp_projection=None,
    client=None,
    remote_k=None,
    remote_img_wh=None,
    remote_frames=2,
) -> dict:
    """Run a single evaluation episode. Returns dict with stats.

    client: if set, a ficelle PolicyClient replaces local policy inference at
    CHUNK granularity (cartesian sync mode only) — see the remote branch in
    the tick loop below. remote_k/remote_img_wh configure it (see main())."""
    n_clamped = 0
    # Remote mode's local action queue (chunk from the last client.infer(),
    # drained one action per tick — mirrors the local policy's own internal
    # action queue, just kept client-side instead of inside `policy`).
    action_queue = []
    # Gripper close-latch (sync path). The gripper is an ABSOLUTE position
    # target re-sent every cycle; successive diffusion draws disagree, so the
    # servo chases a dancing target: visible hesitation, and the deep close
    # commands never persist long enough for the fingers to physically get
    # there (they sit near the time-average of the oscillation). Once any
    # command crosses the threshold, hold the running max — legitimate for
    # end-at-lift datasets where no demo reopens near the object.
    grip_latch = None
    latched_at_step = None
    n_rejected = 0  # arm commands refused by the server (IK-jump watchdog etc.)
    dt = 1.0 / fps
    episode_start = time.perf_counter()
    if dump_dir is not None:
        dump_dir = _Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
    # Training resolution for local inference (remote mode resizes to the
    # server-advertised remote_img_wh in its own branch below).
    img_hw = _policy_img_hw(policy) if client is None else None
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
            grasp_projection=grasp_projection,
        )
        recv_ms = time.perf_counter() * 1000.0
        lat_min_offset = min(lat_min_offset, recv_ms - frame_ts_ms)
        frame_staleness = (recv_ms - frame_ts_ms) - lat_min_offset
        frame_dts = (frame_ts_ms - lat_prev_ts) if lat_prev_ts is not None else None
        lat_prev_ts = frame_ts_ms

        # Resize to training res BEFORE the dump so it stays the exact policy input.
        if img_hw is not None and camera_image.shape[:2] != img_hw:
            camera_image = cv2.resize(camera_image, img_hw[::-1],
                                      interpolation=cv2.INTER_AREA)

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

        if client is None:
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
        if client is None:
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
                if client is None and skip_stale and loop_periods:
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
                if client is not None:
                    # Remote CHUNK-granularity inference: refill the local
                    # action queue with one ficelle round trip when it runs
                    # dry, then pop one action per tick — the wire-protocol
                    # equivalent of the local policy's own internal action
                    # queue (policy.select_action, below).
                    if not action_queue:
                        # This path builds its own observation instead of going
                        # through build_observation(), so the projection has to be
                        # applied HERE too. Sending raw angles to a policy trained
                        # on (strategy, closure) is silent — no error, just a state
                        # channel out of distribution.
                        def _state(g):
                            if grasp_projection is None:
                                return g
                            s_, c_ = grasp_projection.encode(float(g[0]), float(g[1]))
                            return [0.0 if math.isnan(s_) else s_, c_]

                        if remote_frames == 2:
                            (img_prev, grip_prev, _ts_prev), (img_now, grip_now, _ts_now) = camera.get_pair()
                            img_prev_r = cv2.resize(img_prev, remote_img_wh, interpolation=cv2.INTER_AREA)
                            img_now_r = cv2.resize(img_now, remote_img_wh, interpolation=cv2.INTER_AREA)
                            obs = {
                                "observation.images.cam0": np.stack([img_prev_r, img_now_r]),
                                "observation.state": np.stack(
                                    [_state(grip_prev), _state(grip_now)]).astype(np.float32),
                                "task": task,
                            }
                        else:
                            # Single-frame policies (Pi0/Pi0.5): freshest frame
                            # at replan time, unstacked wire shapes.
                            img_now, grip_now, _ts_now = camera.get()
                            img_now_r = cv2.resize(img_now, remote_img_wh, interpolation=cv2.INTER_AREA)
                            obs = {
                                "observation.images.cam0": img_now_r,
                                "observation.state": np.asarray(_state(grip_now), dtype=np.float32),
                                "task": task,
                            }
                        reply = client.infer(obs)
                        action_queue.extend(list(reply["actions"][:remote_k]))
                    a_np = action_queue.pop(0)
                else:
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
        gg1 = float(gripper_goal[0])
        gg2 = float(gripper_goal[1]) if len(gripper_goal) > 1 else 0.0
        if grasp_projection is not None:
            # The policy's last two outputs are (strategy, closure), not angles.
            # Decode to angles HERE, before latch/gain/assist, so everything
            # downstream keeps operating on angles exactly as it always has.
            #
            # closure = 1 means "drive fully closed along this strategy" — the
            # object stops the fingers at the torque cap. That is the whole point:
            # the policy no longer has to predict an object-dependent angle.
            _sc = (gg1, gg2)
            gg1, gg2 = grasp_projection.decode(gg1, gg2)
            gg1, gg2, _clamped = clamp_to_command_limits(gg1, gg2)
            if step == 0 or _sc[1] >= 0.999:
                print(f"proj step {step:3d} | s={_sc[0]:.3f} c={_sc[1]:.3f} -> "
                      f"prox={math.degrees(gg1):.1f}° dist={math.degrees(gg2):.1f}°"
                      f"{' CLAMPED' if _clamped else ''}", flush=True)
        if grip_assist is not None:
            # Minimal intervention: the assist only acts once the policy has
            # settled closed WITHOUT load (see GripAssist). Outside that state
            # the policy's gripper command passes through untouched, so the
            # approach and any already-good grasp are never perturbed.
            _load = camera.get_load()
            _model_grip = (gg1, gg2)
            gg1, gg2 = grip_assist.update((gg1, gg2), _load, step,
                                          meas=tuple(state[-2:]))
            if assist_log is not None:
                # Structured per-tick trace: everything needed to reconstruct a
                # grasp attempt offline (why it engaged or didn't, what it sent,
                # what the fingers and the load actually did).
                # Re-opened in append mode per tick ON PURPOSE: each line is
                # flushed and closed immediately, so the trace survives the run
                # being killed — which is how these runs usually end. The cost
                # (tens of microseconds) is noise next to the loop's gRPC round
                # trips, and this is opt-in debug output.
                with open(assist_log, "a") as f:
                    f.write(_json.dumps({
                        "step": step, "t": time.perf_counter() - episode_start,
                        "model_grip": [round(float(v), 4) for v in _model_grip],
                        "sent_grip": [round(float(gg1), 4), round(float(gg2), 4)],
                        "obs_grip": [round(float(v), 4) for v in state[-2:]],
                        "load": [round(float(v), 1) for v in _load],
                        "state": grip_assist.state,
                        "offset": round(grip_assist.offset, 4),
                        "lag": round(float(grip_assist.lag), 5),
                        "advance": round(float(grip_assist.advance), 5),
                        "why": grip_assist.why,
                    }) + "\n")
        else:
            if latch_close is not None:
                if grip_latch is None and max(gg1, gg2) > latch_close:
                    grip_latch = [gg1, gg2]
                    latched_at_step = step
                    print(f"CLOSE LATCHED at step {step}", flush=True)
                if grip_latch is not None:
                    grip_latch = [max(grip_latch[0], gg1), max(grip_latch[1], gg2)]
                    gg1, gg2 = grip_latch
            gg1, gg2 = apply_grip_gain(gg1, gg2, grip_gain, grip_ref)
        gripper_stub.SendMotorCommand(
            gripper_pb2.MotorCommand(
                motor1_goal=gg1, motor2_goal=gg2,
                motor1_torque_limit=grip_torque_limit,
                motor2_torque_limit=grip_torque_limit,
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
            obs_dist = obs_g[1] if len(obs_g) > 1 else 0.0
            model_dist = gripper_goal[1] if len(gripper_goal) > 1 else 0.0
            load1, load2 = camera.get_load()
            # `cmd` is what was actually SENT (gg1,gg2 — post commit/latch/gain
            # override); the model's raw goal follows in parens so an assist
            # override is visible (cmd pinned while the model's own goal drifts).
            assist_tag = ""
            if grip_assist is not None:
                assist_tag = (f" | {grip_assist.state}+{grip_assist.offset:.3f}"
                              f" lag={grip_assist.lag:.4f} adv={grip_assist.advance:.4f}"
                              f" [{grip_assist.why}]")
            # With the projection active, state[-2:] and the model's own output are
            # (strategy, closure), NOT angles — labelling them prox/dist made a
            # firm 60 deg grasp read as "prox=+0.02", which is badly misleading.
            # Print the pair under its real names and the angles it means.
            if grasp_projection is not None:
                _op, _od = grasp_projection.decode(float(obs_g[0]), float(obs_dist))
                obs_txt = (f" | obs: s={obs_g[0]:+.4f} c={obs_dist:+.4f}"
                           f" (= prox {math.degrees(_op):5.1f}° dist {math.degrees(_od):5.1f}°)")
                model_txt = f" (model s={gripper_goal[0]:+.3f} c={model_dist:+.3f})"
            else:
                obs_txt = f" | obs: prox={obs_g[0]:+.4f} dist={obs_dist:+.4f}"
                model_txt = f" (model {gripper_goal[0]:+.3f}/{model_dist:+.3f})"
            print(
                f"step {step:3d} | gripper cmd: prox={gg1:+.4f} dist={gg2:+.4f}"
                f"{model_txt}"
                f"{assist_tag}"
                f"{obs_txt}"
                f" | load: prox={load1:+.0f} dist={load2:+.0f}",
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
            debug_display(cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))

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
            f"episode — the executed motion differs from what the policy "
            f"commanded. See the 'ARM REJECTED' lines above for the reason "
            f"per command (IK-jump watchdog, contact/target-lead cap, ...); "
            f"they are different faults with different fixes.",
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
    latch_close=None,
    commit_close=None,
    log_deltas=False,
    log_latency=False,
    dump_dir=None,
    grip_gain=1.0,
    grip_ref=(0.0, 0.0),
    grip_torque_limit=0.0,
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
    img_hw = _policy_img_hw(policy)  # training resolution, see _policy_img_hw

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
                             start_rot=ep_start_rot.astype(np.float64),
                             latch_close=latch_close,
                             grip_gain=grip_gain, grip_ref=grip_ref,
                             grip_torque_limit=grip_torque_limit)
    episode_start = time.perf_counter()
    lat_min_offset = float("inf")
    stats = {"infer": [], "skip": [], "chunk": []}
    cycle = 0
    first_cycle = True
    success = False
    n_commits = 0
    first_commit_tick = None
    try:
        while executor.sent_count < max_steps and executor.frozen is None:
            (img_prev, grip_prev, _ts_prev), (img_now, grip_now, ts_now) = camera.get_pair()
            if img_hw is not None and img_now.shape[:2] != img_hw:
                img_prev = cv2.resize(img_prev, img_hw[::-1], interpolation=cv2.INTER_AREA)
                img_now = cv2.resize(img_now, img_hw[::-1], interpolation=cv2.INTER_AREA)
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

            # Close-commit: the model initiates a close almost only by
            # CONTINUING one it observes (finger motion / raised grip state) —
            # measured P(initiate|static pre-grasp) ~ 0-15% vs 100% once the
            # state shows a started close. Replanning every ~3 ticks re-rolls
            # that dice and executes only chunk heads, so the close stays
            # forever in the receding tail. When a drawn chunk crosses the
            # threshold, execute it TO THE END without replanning: the fingers
            # actually move, the state rises, and the next replan continues
            # the close instead of re-deciding it.
            # Trigger on a crossing ANYWHERE in the chunk. A head-only window
            # (tried: first 8 actions) sounds cleaner — "commit only when the
            # close is imminent" — but it silently re-creates the
            # never-ignites failure for models that schedule their close
            # further ahead than the window (measured: the mustard-200 model
            # kept its close 9-13 steps out through 40+ replans; the commit
            # never armed and the arm pressed into the workspace boundary).
            # The cost of the full-chunk trigger is an occasional early
            # commit that closes while still advancing (a push, sometimes a
            # miss); the cost of the window is never grasping at all.
            if commit_close is not None and any(
                float(max(a[9], a[10] if a.shape[0] > 10 else 0.0)) > commit_close
                for a in chunk[k:]
            ):
                if n_commits == 0:
                    first_commit_tick = executor.sent_count
                n_commits += 1
                print(f"CLOSE COMMIT at tick {executor.sent_count}: chunk crosses "
                      f"{commit_close:.2f}, executing to the end without replan", flush=True)
                while (executor.remaining() > 0 and executor.frozen is None
                       and executor.sent_count < max_steps):
                    time.sleep(0.005)

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
            f"clamped {executor.n_clamped}"
            + (f" | CLOSE LATCHED at tick {executor.latched_at_tick}"
               if executor.latched_at_tick is not None else "")
            + (f" | close commits {n_commits} (first at tick {first_commit_tick})"
               if n_commits else ""),
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
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested (--device cuda) but unavailable — falling back to CPU.")
        device = torch.device("cpu")

    remote = args.policy_addr is not None
    if remote and args.async_exec:
        raise SystemExit("--policy_addr supports sync mode only (not --async_exec).")

    # Grasp projection: does this policy emit (strategy, closure) or angles?
    # Getting this wrong is silent in both directions — a closure of 1.0 sent
    # as radians barely moves the gripper; an angle of 0.9 rad read as a
    # closure decodes to a near-full close every step — so the decision is
    # logged unconditionally.
    projection = None
    if args.grasp_projection == "on":
        projection = GraspProjection()
    elif args.grasp_projection == "auto":
        detected = detect_grasp_projection(args.checkpoint)
        if detected:
            projection = GraspProjection()
        elif detected is None:
            # REFUSE rather than guess. Assuming raw is the silent failure:
            # a projected policy's closure of 1.0 would be sent as 1.0 radian
            # (57 deg proximal), i.e. a partial close — the exact under-close
            # this projection exists to fix, with nothing in the log to say so.
            raise SystemExit(
                "Could not determine whether this policy was trained on the "
                "grasp projection"
                + (" (remote inference — there is no local checkpoint to "
                   "inspect)" if not args.checkpoint
                   else " (no readable normaliser stats)") + ".\n"
                "Refusing to guess: assuming raw angles would silently send a "
                "closure of 1.0 as 1.0 RADIAN and under-close every grasp.\n"
                "Pass --grasp_projection on (projected dataset) or off (raw "
                "joint angles) explicitly."
            )
    logger.info(
        f"Gripper action space: "
        f"{'(strategy, closure) -> decoded to angles' if projection else 'raw angles'}"
    )

    policy = preprocessor = postprocessor = None
    client = remote_k = remote_img_wh = None
    remote_frames = 2

    if remote:
        # Lazy import: the whole point of remote mode is that the client
        # machine needs no checkpoint (and, on a bare robot box, no ficelle
        # client either unless this flag is actually used). open_client
        # dispatches on the address shape: an iroh ticket (starts with
        # "endpoint") gets IrohPolicyClient, anything else (HOST:PORT) gets
        # PolicyClient (websocket) — so --policy_addr transparently accepts
        # either, unchanged from the caller's point of view.
        try:
            from ficelle_client import open_client
        except ImportError as e:
            raise SystemExit(
                "--policy_addr requires the ficelle_client package (uv pip install ./ficelle/client)"
            ) from e
        client_kw = {"jpeg_quality": args.jpeg_quality, "resize": args.resize}
        if args.policy_addr.startswith("endpoint") and ":" not in args.policy_addr:
            # iroh's default infer timeout (5 s) is tighter than a cold
            # server's FIRST inference (model warm-up on the big VLA
            # checkpoints: measured 30+ s for pi05 before the kernels are hot).
            client_kw["infer_timeout"] = 60.0
        client = open_client(args.policy_addr, **client_kw)
        metadata = client.metadata
        state_spec = metadata.get("observations", {}).get("observation.state", {})
        if (metadata.get("action_dim") != 11
                or list(state_spec.get("frame_shape", [])) != [2]
                or metadata.get("n_obs_steps") not in (1, 2)):
            raise SystemExit(
                f"--policy_addr server at {args.policy_addr} reports action_dim="
                f"{metadata.get('action_dim')}, state frame_shape="
                f"{state_spec.get('frame_shape')}, n_obs_steps={metadata.get('n_obs_steps')} "
                "— this eval only supports 11D-cartesian/2D-gripper-state policies "
                "with n_obs_steps 1 (Pi0/Pi0.5 single frame) or 2 (Diffusion pair)."
            )
        joint_mode = False
        use_relative_proprio = False
        # 1 = single-frame policies (Pi0/Pi0.5): unstacked wire shapes;
        # 2 = frame-history policies (Diffusion): stacked (2, ...) arrays.
        remote_frames = int(metadata["n_obs_steps"])
        server_n_action_steps = metadata["n_action_steps"]
        remote_k = (min(args.n_action_steps, server_n_action_steps)
                    if args.n_action_steps is not None else server_n_action_steps)
        h, w, _c = metadata["observations"]["observation.images.cam0"]["frame_shape"]
        remote_img_wh = (w, h)
        logger.info(
            f"Remote policy: {metadata.get('policy_type')} from {metadata.get('checkpoint')} "
            f"@ {args.policy_addr} | n_action_steps used={remote_k} | "
            f"image wire shape {(remote_frames, h, w, 3) if remote_frames > 1 else (h, w, 3)}"
        )
    else:
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
        # The preprocessor's DeviceProcessorStep is restored from the checkpoint's
        # saved processor config (device baked in as "cuda"), which overrides
        # policy.config.device. Override it explicitly so observations land on the
        # same device as the weights — otherwise --device cpu crashes with an
        # input(cuda)/weight(cpu) mismatch. (The postprocessor already targets cpu.)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            pretrained_path=args.checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

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

    # ---- SAFETY: never leave the gripper clamped on exit ----------------
    # The servo holds its last commanded goal, so a run that ends while the
    # gripper is closed (normal end, Ctrl-C, or a crash) leaves it squeezing at
    # the torque cap indefinitely. Observed on the real gripper 2026-07-28:
    # found at prox 1.387 with load pinned at 250 long after a killed run —
    # hard on the motor, and the NEXT episode then starts with the policy
    # observing a CLOSED gripper, which is out of distribution (every demo
    # starts open) and makes it predict lift/hold instead of approach/close.
    # atexit covers normal return, KeyboardInterrupt and exceptions; a SIGKILL
    # (kill -9) cannot be caught, so after one of those, reopen manually.
    # Ordering matters: the reopen is an RPC, so it must happen BEFORE the
    # channels are closed. Hence ONE idempotent teardown that reopens and then
    # releases resources, called explicitly at the end of main() and registered
    # with atexit for the abnormal paths.
    def _teardown():
        if getattr(_teardown, "done", False):
            return
        _teardown.done = True
        try:
            gripper_stub.SendMotorCommand(gripper_pb2.MotorCommand(
                motor1_goal=float(args.start_gripper[0]),
                motor2_goal=float(args.start_gripper[1]),
                motor1_torque_limit=float(args.grip_torque_limit),
                motor2_torque_limit=float(args.grip_torque_limit),
            ))
            print(f"gripper reopened to {tuple(args.start_gripper)} on exit", flush=True)
        except Exception as e:  # noqa: BLE001 — best-effort teardown
            print(f"WARNING: could not reopen the gripper on exit: {e}", flush=True)
        for close in (camera.stop, arm_channel.close, gripper_channel.close):
            try:
                close()
            except Exception:  # noqa: BLE001 — already going away
                pass

    atexit.register(_teardown)

    # ---- Evaluation loop ----
    results = []
    logger.info(
        f"\nStarting evaluation: {args.num_episodes} episodes, "
        f"max {args.max_steps} steps/episode at {args.fps} Hz\n"
    )

    for ep in range(args.num_episodes):
        # Reset environment with randomization. --home_joints overrides the
        # server's built-in home: with relative (camera-local delta) policies
        # the start pose anchors the entire trajectory, so it must reproduce
        # the demos' start camera view. --no_reset skips the move entirely
        # (hand-placed start; the server resynced its integrator at torque-on).
        if args.no_reset:
            logger.info("--no_reset: starting from the arm's current pose")
        else:
            reset_req = arm_pb2.ResetRequest()
            if args.home_joints is not None:
                reset_req.joint_positions.extend(args.home_joints)
            reset_resp = arm_stub.Reset(reset_req)
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

        # Reset policy action queue (remote mode keeps its queue client-side —
        # nothing to reset here, run_episode starts each episode with a fresh
        # local action_queue).
        if policy is not None:
            policy.reset()

        # Small delay for physics to settle (and for gripper to actually open)
        time.sleep(0.5)

        # Capture start pose for relative proprioception (after reset)
        start_pos, start_rot = None, None
        if use_relative_proprio:
            start_pos, start_rot = capture_start_pose(arm_stub, arm_pb2)

        if args.no_reset:
            logger.info(f"Episode {ep + 1}/{args.num_episodes} — from current arm pose")
        else:
            logger.info(
                f"Episode {ep + 1}/{args.num_episodes} — "
                f"cube at ({reset_resp.cube_x:.3f}, {reset_resp.cube_y:.3f}, {reset_resp.cube_z:.3f})"
            )

        # Fresh assist state per EPISODE — it latches a grasp and tracks
        # arm/disarm across ticks, so it must not leak between episodes.
        assist = (GripAssist(
            args.grip_assist, ref=tuple(args.start_gripper),
            min_close=args.assist_min_close,
            stable_ticks=args.assist_stable_ticks,
            stable_eps=args.assist_stable_eps,
            step=args.assist_step, max_extra=args.assist_max_extra,
            dwell_ticks=args.assist_dwell_ticks,
            confirm_ticks=args.assist_confirm_ticks,
            lag=args.assist_lag, settle_eps=args.assist_settle_eps,
            squeeze=args.assist_squeeze,
        ) if args.grip_assist is not None else None)

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
                latch_close=args.latch_close,
                commit_close=args.commit_close,
                grip_gain=args.grip_gain,
                grip_ref=tuple(args.start_gripper),
                grip_torque_limit=args.grip_torque_limit,
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
            grip_gain=args.grip_gain,
            grip_ref=tuple(args.start_gripper),
            grip_torque_limit=args.grip_torque_limit,
            latch_close=args.latch_close,
            grip_assist=assist,
            grasp_projection=projection,
            assist_log=args.assist_log,
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
            client=client,
            remote_k=remote_k,
            remote_img_wh=remote_img_wh,
            remote_frames=remote_frames,
        )
        if assist is not None:
            # Per-episode assist bookkeeping: did it engage, did it find
            # contact, and how much extra closure the policy was short by
            # (offset = the measured under-close, useful for A/B analysis).
            print(f"GRIP ASSIST | final state {assist.state} | extra closure "
                  f"{assist.offset:+.3f} rad | contacts {assist.n_gripped} | "
                  f"empty top-ups {assist.n_exhausted} | first engaged at step "
                  f"{assist.trigger_step}", flush=True)

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

    if args.debug and _DEBUG_GUI:
        cv2.destroyAllWindows()
    # Reopens the gripper, then stops the camera and closes the channels (also
    # registered with atexit, and idempotent, so abnormal exits are covered).
    _teardown()


if __name__ == "__main__":
    main()
