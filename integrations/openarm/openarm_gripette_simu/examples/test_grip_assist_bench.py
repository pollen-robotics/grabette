"""Closed-loop bench test of --grip_assist against the REAL gripper. No arm.

Validates the assist's stall detection with real load/position signals before
it runs on the arm, where a mis-tuned threshold would masquerade as a policy
failure. It stands in for the policy with a command that deliberately
UNDER-CLOSES — the failure the assist exists to fix — then drives GripAssist's
real update() loop at eval rate and reports whether contact was found, how much
extra closure it took (= the under-close magnitude), and the lag/load trace.

Run it twice: with the object in the jaws (expect GRIPPED) and with them clear
(expect EXHAUSTED, i.e. no false grasp).

  uv run python examples/test_grip_assist_bench.py \\
      --gripper_addr <pi-ip>:50051 --under_close 0.12 --label mustard
"""
import argparse
import importlib.util
import time
from pathlib import Path

import grpc
import numpy as np

from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc

# Import GripAssist from evaluate.py (same directory) — the REAL class.
_spec = importlib.util.spec_from_file_location(
    "_ev", str(Path(__file__).with_name("evaluate.py")))
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)
GripAssist = _ev.GripAssist


def main():
    p = argparse.ArgumentParser(description="Bench test --grip_assist on real hardware")
    p.add_argument("--gripper_addr", default="localhost:50051")
    p.add_argument("--label", default="trial")
    p.add_argument("--torque", type=float, default=0.25)
    p.add_argument("--ref", type=float, nargs=2, default=[0.0, 0.0],
                   metavar=("PROX", "DIST"), help="Open reference (like --start_gripper)")
    p.add_argument("--probe_to", type=float, nargs=2, default=[1.2, 1.2],
                   metavar=("PROX", "DIST"),
                   help="Direction of the stand-in policy's close (posture)")
    p.add_argument("--probe_from", type=float, default=0.30,
                   help="Travel at which to start probing for contact. Must clear the open "
                        "pose, which reads as a stall itself (fingers resting off the "
                        "commanded 0, servo pushing into the open stop).")
    p.add_argument("--probe_dwell", type=float, default=0.45,
                   help="Seconds to hold each probe step before judging contact. Must be "
                        "long enough for the fingers to ARRIVE — a short dwell reads motion "
                        "lag/load and reports contact on the first step.")
    p.add_argument("--under_close", type=float, default=0.12,
                   help="How far SHORT of contact the stand-in policy stops (rad of "
                        "travel). The assist must make up this much.")
    p.add_argument("--fps", type=float, default=10.0, help="Eval loop rate to emulate")
    p.add_argument("--max_ticks", type=int, default=120)
    # assist tunables (defaults = the measured recommendation)
    p.add_argument("--load_thresh", type=float, default=50.0)
    p.add_argument("--assist_lag", type=float, default=0.010)
    p.add_argument("--assist_step", type=float, default=0.02)
    p.add_argument("--assist_max_extra", type=float, default=0.4)
    p.add_argument("--assist_dwell_ticks", type=int, default=2)
    p.add_argument("--assist_confirm_ticks", type=int, default=2)
    p.add_argument("--assist_stable_ticks", type=int, default=3)
    args = p.parse_args()

    stub = gripper_pb2_grpc.GripperServiceStub(grpc.insecure_channel(args.gripper_addr))
    stub.Ping(gripper_pb2.PingRequest())
    stub.SetTorque(gripper_pb2.TorqueCommand(enable=True))   # ONCE, never per goal

    def send(goal):
        r = stub.SendMotorCommand(gripper_pb2.MotorCommand(
            motor1_goal=float(goal[0]), motor2_goal=float(goal[1]),
            motor1_torque_limit=args.torque, motor2_torque_limit=args.torque))
        if not r.success:
            raise RuntimeError(f"rejected {goal}: {r.error}")

    def read():
        ms = stub.ReadMotors(gripper_pb2.ReadMotorsRequest())
        return ((ms.motor1_position, ms.motor2_position),
                (ms.motor1_load, ms.motor2_load))

    ref = np.asarray(args.ref, float)
    direction = np.asarray(args.probe_to, float) - ref
    direction /= np.linalg.norm(direction)

    try:
        # 1) Find where the object actually stops the fingers, so the stand-in
        #    policy can be placed a known distance SHORT of it (that is the
        #    under-close we are asking the assist to recover).
        print("probing for contact to place the stand-in policy command...")
        send(ref); time.sleep(1.0)
        contact_travel = None
        # Start past the open pose: commanding the open reference while the
        # fingers rest slightly off it IS a stall (persistent lag + the servo
        # pushing into the open stop, measured |load| up to ~96), so probing
        # from 0 reports contact immediately. The assist never sees this because
        # it only acts once the command is >= min_close from the reference.
        travel = args.probe_from
        # NOTE: dwell long enough for the fingers to actually ARRIVE before
        # judging. A short dwell reads motion lag + motion load and reports
        # "contact" on the very first step (the trap this script fell into).
        while travel < float(np.linalg.norm(np.asarray(args.probe_to, float) - ref)):
            goal = ref + direction * travel
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < args.probe_dwell:
                send(goal); time.sleep(0.02)
            pos, load = read()
            lag = float(np.max(np.abs(goal - np.asarray(pos))))
            if lag > args.assist_lag and max(abs(load[0]), abs(load[1])) >= args.load_thresh:
                contact_travel = travel
                break
            travel += 0.02
        send(ref); time.sleep(1.0)
        if contact_travel is None:
            print("  no contact found on the probe — jaws are clear (that is fine for "
                  "the empty run; the policy command will simply be a mid-range close)")
            contact_travel = 0.6
        else:
            print(f"  contact at travel {contact_travel:.3f}")

        policy_travel = contact_travel - args.under_close
        if policy_travel < 0.20:
            # A stand-in command that barely leaves the open pose is not a
            # "close" at all: the assist would (correctly) never engage, and the
            # run would look like a failure of the assist rather than of the
            # setup. Refuse instead of producing a meaningless result.
            raise SystemExit(
                f"probe put contact at travel {contact_travel:.3f}, so a "
                f"{args.under_close:.3f} under-close leaves only "
                f"{policy_travel:.3f} of closure — too little to count as a "
                f"close.\nPlace the object deeper in the jaws, or reduce "
                f"--under_close. (If contact was reported implausibly early, "
                f"raise --probe_dwell: a short dwell reads motion lag as contact.)")
        policy_cmd = tuple(ref + direction * policy_travel)
        print(f"stand-in policy will hold ({policy_cmd[0]:+.3f}, {policy_cmd[1]:+.3f}) "
              f"= {args.under_close:.3f} rad short of contact\n")

        # 2) Run the REAL assist loop at eval rate.
        assist = GripAssist(
            args.load_thresh, ref=tuple(args.ref), min_close=0.15,
            stable_ticks=args.assist_stable_ticks, stable_eps=0.01,
            step=args.assist_step, max_extra=args.assist_max_extra,
            dwell_ticks=args.assist_dwell_ticks,
            confirm_ticks=args.assist_confirm_ticks, lag=args.assist_lag)
        dt = 1.0 / args.fps
        print(f"{'tick':>4} {'state':>10} {'offset':>7} {'sent_p':>7} {'meas_p':>7} "
              f"{'lag':>7} {'load_p':>7}")
        for tick in range(args.max_ticks):
            pos, load = read()
            out = assist.update(policy_cmd, load, step=tick, meas=pos)
            send(out)
            if tick % 4 == 0 or assist.state in ("GRIPPED", "EXHAUSTED"):
                print(f"{tick:>4} {assist.state:>10} {assist.offset:7.3f} {out[0]:7.3f} "
                      f"{pos[0]:7.3f} {getattr(assist,'lag',float('nan')):7.4f} "
                      f"{load[0]:7.0f}")
            if assist.state in ("GRIPPED", "EXHAUSTED"):
                break
            time.sleep(dt)

        print(f"\n[{args.label}] RESULT: {assist.state} | extra closure "
              f"{assist.offset:+.3f} rad (asked for {args.under_close:.3f}) | "
              f"contacts {assist.n_gripped} | empty top-ups {assist.n_exhausted}")
        if assist.state == "GRIPPED":
            print("  -> assist recovered the under-close and found the object")
        elif assist.state == "EXHAUSTED":
            print("  -> no contact within max_extra (correct if the jaws were clear; "
                  "if an object WAS present, raise --assist_max_extra or lower "
                  "--assist_lag/--load_thresh)")
        else:
            print("  -> never resolved: check --assist_stable_ticks (the stand-in "
                  "command is constant, so it should settle immediately)")
    finally:
        send(args.ref); time.sleep(1.0)
        stub.SetTorque(gripper_pb2.TorqueCommand(enable=False))
        print("gripper reopened, torque released")


if __name__ == "__main__":
    main()
