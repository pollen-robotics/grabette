#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Adapted from openarm-can-zero-position-calibration (Enactic, Inc., Apache 2.0).
# Modifications: removed all gripper motor I/O so it works on Gripette-equipped
# OpenArms where the gripper is NOT on the same CAN bus (it's a separate Pi
# service). The original script's init_gripper_motor(0x08, 0x18) call blocks /
# misbehaves when CAN ID 0x08 doesn't answer.
#
# Apart from gripper removal, the calibration sequence is IDENTICAL to the
# upstream tool: the same bump-to-stop motions, the same step sizes and gains,
# the same ideal-zero positions per joint, the same per-joint sign convention.
#
# Usage (right arm):
#   sudo ip link set can0 up type can bitrate 1000000
#   uv run python examples/openarm_gripette/calibrate_arm_no_gripper.py \
#       --canport can0 --arm-side right_arm
#
# Before running:
#   - Disconnect the Gripette (or just leave it; this script doesn't touch it).
#   - Make sure the arm has clearance to swing into every mechanical stop.
#   - Have an e-stop within reach. The script bumps joints into hard limits.

import argparse
import time
from enum import IntEnum

import numpy as np
import openarm_can as oa

# ---------- IDs / Limits / Signs ----------


class JointID(IntEnum):
    J1 = 0
    J2 = 1
    J3 = 2
    J4 = 3
    J5 = 4
    J6 = 5
    J7 = 6


mech_lim = {
    JointID.J1: [np.deg2rad(-80),  np.deg2rad(200)],
    JointID.J2: [np.deg2rad(-100), np.deg2rad(100)],
    JointID.J3: [np.deg2rad(-90),  np.deg2rad(90)],
    JointID.J4: [np.deg2rad(0),    np.deg2rad(140)],
    JointID.J5: [np.deg2rad(-90),  np.deg2rad(90)],
    JointID.J6: [np.deg2rad(-45),  np.deg2rad(45)],
    JointID.J7: [np.deg2rad(-90),  np.deg2rad(90)],
}

# Per-joint sign convention used in ideal-zero delta computation
JOINT_SIGN = {
    JointID.J1: -1.0, JointID.J2: -1.0, JointID.J3: +1.0, JointID.J4: +1.0,
    JointID.J5: +1.0, JointID.J6: +1.0, JointID.J7: +1.0,
}

# ---------- Core motion primitives ----------


def interpolate(openarm, comp, joint_id: JointID, delta_rad,
                kp=52.0, kd=1.5, torque_assist=0.0, interp_time=2.0):
    """Linear interp: current -> current+delta with MIT control."""
    idx = int(joint_id)
    m = comp.get_motors()[idx]
    q0 = m.get_position()
    q1 = q0 + float(delta_rad)
    n_steps, dt = 500, (interp_time / 500.0)
    tau = np.copysign(1.0, q1 - q0) * abs(torque_assist)
    for i in range(n_steps + 1):
        alpha = i / n_steps
        q = q0 + (q1 - q0) * alpha
        comp.mit_control_one(idx, oa.MITParam(kp, kd, q, 0.0, tau))
        openarm.recv_all()
        time.sleep(dt)


def _hit_thresholds(idx):
    """Arm-only thresholds (gripper case removed from upstream)."""
    if idx == 0:       # J1 on arm is stricter
        return 0.0125, 5.0
    return 0.1, 2.0


def bump_to_limit(openarm, comp, joint_id,
                  step_deg=0.2, kp=45.0, kd=1.2, torque_bias=0.0):
    """Step until mechanical stop. Return traveled delta [rad]."""
    idx = int(joint_id)
    step_rad = np.deg2rad(step_deg)
    tau_cmd = np.copysign(1.0, step_deg) * abs(torque_bias)

    motors = comp.get_motors()
    q_start = motors[idx].get_position()
    q_target = q_start

    dq_th, tau_th = _hit_thresholds(idx)

    while True:
        q_target += step_rad
        comp.mit_control_one(idx, oa.MITParam(kp, kd, q_target, 0.0, tau_cmd))
        openarm.recv_all()
        time.sleep(0.005)

        m = comp.get_motors()[idx]
        if np.abs(m.get_velocity()) < dq_th and np.abs(m.get_torque()) > tau_th:
            delta_rad = m.get_position() - q_start
            delta_deg = np.rad2deg(delta_rad)
            print(f"[INFO] mechanical stop (Joint {joint_id.name}): "
                  f"{delta_rad:.4f} rad / {delta_deg:.2f}°")
            return float(delta_rad)


def calc_delta_to_zero_pos_joint(initial_rad: float,
                                 ideal_limit_rad: float,
                                 delta_to_stop_rad: float,
                                 joint_id: JointID) -> float:
    """Relative delta from 'hit position' to 'ideal limit', with per-joint sign."""
    q_hit = initial_rad + delta_to_stop_rad
    delta_to_ideal = ideal_limit_rad - q_hit
    return float(JOINT_SIGN.get(joint_id, 1.0) * delta_to_ideal)


# ---------- One function per arm-side to keep intentions explicit ----------


def _run_right_sequence(openarm, arm):
    """Identical to upstream _run_right_sequence minus the gripper bump."""
    d_j4 = bump_to_limit(openarm, arm,  JointID.J4, step_deg=-0.2)
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J2, np.deg2rad(5),  interp_time=0.4)
    time.sleep(0.5)
    d_j3 = bump_to_limit(openarm, arm,  JointID.J3)
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J3,
                -mech_lim[JointID.J3][1], interp_time=1.0)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J2, -np.deg2rad(5), interp_time=0.4)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J4,  np.pi / 2.0)
    time.sleep(0.5)

    d_j5 = bump_to_limit(openarm, arm,  JointID.J5)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J5, -mech_lim[JointID.J5][1])
    time.sleep(0.5)

    d_j6 = bump_to_limit(openarm, arm,  JointID.J6)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J6, -mech_lim[JointID.J6][1])
    time.sleep(0.5)

    d_j7 = bump_to_limit(openarm, arm,  JointID.J7)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J7, -mech_lim[JointID.J7][1])
    time.sleep(0.5)

    d_j2 = bump_to_limit(openarm, arm,  JointID.J2, step_deg=-0.2)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J2, np.deg2rad(10),
                interp_time=0.9, kp=180, kd=2.0)
    time.sleep(0.5)

    d_j1 = bump_to_limit(openarm, arm,  JointID.J1,
                         step_deg=-0.2, kp=180, kd=2.1)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J1, np.deg2rad(80))
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J4, -np.pi / 2.0)
    time.sleep(2.5)

    ideal = {
        JointID.J7: np.pi / 2.0,
        JointID.J6: np.pi / 4.0,
        JointID.J5: np.pi / 2.0,
        JointID.J4: 0.0,
        JointID.J3: np.pi / 2.0,
        JointID.J2: np.deg2rad(-10),
        JointID.J1: np.deg2rad(-80),
    }
    deltas = [d_j1, d_j2, d_j3, d_j4, d_j5, d_j6, d_j7]
    return ideal, deltas


def _run_left_sequence(openarm, arm):
    """Identical to upstream _run_left_sequence minus the gripper bump."""
    d_j4 = bump_to_limit(openarm, arm,  JointID.J4, step_deg=-0.2)
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J2, -np.deg2rad(5), interp_time=0.4)
    time.sleep(0.5)
    d_j3 = bump_to_limit(openarm, arm,  JointID.J3)
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J3,
                -mech_lim[JointID.J3][1], interp_time=1.0)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J2,  np.deg2rad(5),  interp_time=0.4)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J4,  np.pi / 2.0)
    time.sleep(0.5)

    d_j5 = bump_to_limit(openarm, arm,  JointID.J5)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J5, -mech_lim[JointID.J5][1])
    time.sleep(0.5)

    d_j6 = bump_to_limit(openarm, arm,  JointID.J6)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J6, -mech_lim[JointID.J6][1])
    time.sleep(0.5)

    d_j7 = bump_to_limit(openarm, arm,  JointID.J7)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J7, -mech_lim[JointID.J7][1])
    time.sleep(0.5)

    d_j2 = bump_to_limit(openarm, arm,  JointID.J2)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J2, -np.deg2rad(10),
                interp_time=0.9, kp=180, kd=2.0)
    time.sleep(0.5)

    d_j1 = bump_to_limit(openarm, arm,  JointID.J1, kp=180, kd=2.1)
    time.sleep(0.5)
    interpolate(openarm, arm, JointID.J1, -np.deg2rad(80))
    time.sleep(0.5)

    interpolate(openarm, arm, JointID.J4, -np.pi / 2.0)
    time.sleep(2.5)

    ideal = {
        JointID.J7: np.pi / 2.0,
        JointID.J6: np.pi / 4.0,
        JointID.J5: np.pi / 2.0,
        JointID.J4: 0.0,
        JointID.J3: np.pi / 2.0,
        JointID.J2: np.deg2rad(10),
        JointID.J1: np.deg2rad(80),
    }
    deltas = [d_j1, d_j2, d_j3, d_j4, d_j5, d_j6, d_j7]
    return ideal, deltas


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser(
        description='OpenArm zero-position calibration (7-DOF arm only; no gripper)'
    )
    parser.add_argument('--canport', type=str, default='can0')
    parser.add_argument('--arm-side', type=str,
                        default='right_arm', choices=['right_arm', 'left_arm'])
    args = parser.parse_args()
    print(f"parser arg : {args}")

    # Init: arm motors only — do NOT call init_gripper_motor.
    openarm = oa.OpenArm(args.canport, True)
    openarm.init_arm_motors(
        [oa.MotorType.DM8009, oa.MotorType.DM8009, oa.MotorType.DM4340, oa.MotorType.DM4340,
         oa.MotorType.DM4310, oa.MotorType.DM4310, oa.MotorType.DM4310],
        [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07],
        [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
    )
    openarm.set_callback_mode_all(oa.CallbackMode.STATE)

    print("Enabling...")
    openarm.enable_all()
    time.sleep(0.1)
    print("Enabled...")

    # Hold current pose (same gains as upstream).
    openarm.recv_all()
    arm = openarm.get_arm()
    am = arm.get_motors()

    initial_arm_q = [m.get_position() for m in am]

    arm_params = [
        oa.MITParam(kp, kd, q, 0.0, 0.0)
        for kp, kd, q in zip(
            [300, 300, 150, 150, 40, 40, 30],
            [2.5, 2.5, 2.5, 2.5, 0.8, 0.8, 0.8],
            initial_arm_q,
        )
    ]
    arm.mit_control_all(arm_params)
    openarm.recv_all()

    try:
        # Run side-specific calibration sequence.
        seq = _run_right_sequence if args.arm_side == 'right_arm' else _run_left_sequence
        ideal, deltas = seq(openarm, arm)

        # Compute ideal deltas per joint.
        joint_order = [JointID.J1, JointID.J2, JointID.J3, JointID.J4,
                       JointID.J5, JointID.J6, JointID.J7]
        ideal_delta_map = {}
        for j, d in zip(joint_order, deltas):
            val = calc_delta_to_zero_pos_joint(
                initial_rad=initial_arm_q[int(j)],
                ideal_limit_rad=ideal[j],
                delta_to_stop_rad=d,
                joint_id=j,
            )
            ideal_delta_map[j] = val
            print(f"ideal[{j.name}] : {val:+.6f} rad ({np.rad2deg(val):+.2f}°)")

        # Absolute goal = initial + ideal_delta. Reported for transparency;
        # not commanded (upstream leaves move_to_precise_home as a hook).
        arm_goal_abs = [
            initial_arm_q[i] + ideal_delta_map.get(JointID(i), 0.0)
            for i in range(7)
        ]
        print(f"arm_goal_abs (rad): "
              f"{[round(v, 4) for v in arm_goal_abs]}")

        # Safe disable before writing zeros.
        openarm.disable_all()
        openarm.recv_all()
        openarm.set_zero_all()
        openarm.recv_all()
        print("wrote zero position to arm")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C pressed -> stopping safely")
    finally:
        openarm.disable_all()
        print("[INFO] Motors disabled, exiting safely.")


if __name__ == "__main__":
    main()
