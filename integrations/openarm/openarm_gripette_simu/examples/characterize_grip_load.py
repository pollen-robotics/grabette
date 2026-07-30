"""Measure the Gripette's load-vs-closure curve — sets `--grip_assist LOAD_THRESH`.

GRIPPER ONLY: no arm, no policy. Talks to the Gripette gRPC service (real
hardware, or the sim where load comes from MuJoCo actuator_force — units differ,
never reuse a threshold across them).

WHAT IT MEASURES, AND WHY THIS WAY
`--grip_assist` advances the closure in small increments (--assist_step) and
watches present_load to detect contact. So the number that matters is not the
load of a big slam — it is the load produced by a SMALL press past contact,
which depends on object STIFFNESS:
  - rigid object: a 0.02 rad press reacts hard -> load jumps, easy detection;
  - soft object (e.g. a squeezable bottle): the object just deforms -> weaker
    load, and the finger can creep instead of stalling. This is the WORST CASE
    and the one the threshold must catch.
This tool therefore sweeps the closure in assist-sized steps along a posture
direction, dwelling at each step, and logs the load at each — giving the
load-vs-closure curve per object. Read off: the free-motion floor (from an
'air' trial), where load rises (contact), and how fast it rises after contact.

TWO HARDWARE LESSONS BAKED IN (learned the hard way 2026-07-27):
  1. Torque is enabled ONCE at start, never per command. Enabling torque in the
     same bus tick as a goal makes the servo reset its goal to the present
     position, and the goal is silently dropped.
  2. Goals are re-commanded CONTINUOUSLY, never written once. A single write can
     fail to take on one motor, which looks exactly like "that motor is stuck".

Usage (see the eval README):
  uv run python examples/characterize_grip_load.py \\
      --gripper_addr <pi-ip>:50051 --torque 0.25 \\
      --from 0.0 0.0 --to 1.2 1.2 --step 0.02
Then run one trial per object at the prompt: 'air' first (the floor), then e.g.
'mustard' (soft), 'rigid', 'thin'. Ctrl-C or blank input to finish.
"""
import argparse
import json
import time

import grpc
import numpy as np

from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc


class Gripper:
    """Thin client applying the two hardware lessons above."""

    def __init__(self, addr, torque):
        self._stub = gripper_pb2_grpc.GripperServiceStub(grpc.insecure_channel(addr))
        self._torque = float(torque)
        self._stub.Ping(gripper_pb2.PingRequest())
        # ONCE — never per command (see lesson 1).
        self._stub.SetTorque(gripper_pb2.TorqueCommand(enable=True))

    def read(self):
        ms = self._stub.ReadMotors(gripper_pb2.ReadMotorsRequest())
        return (ms.motor1_position, ms.motor2_position,
                ms.motor1_load, ms.motor2_load)

    def hold(self, goal, seconds, hz=50.0):
        """Command `goal` CONTINUOUSLY for `seconds` (see lesson 2); return the
        samples recorded while holding it."""
        rows, t0, period = [], time.perf_counter(), 1.0 / hz
        while time.perf_counter() - t0 < seconds:
            resp = self._stub.SendMotorCommand(gripper_pb2.MotorCommand(
                motor1_goal=float(goal[0]), motor2_goal=float(goal[1]),
                motor1_torque_limit=self._torque, motor2_torque_limit=self._torque))
            if not resp.success:
                raise RuntimeError(f"gripper rejected goal {goal}: {resp.error}")
            rows.append(self.read())
            time.sleep(period)
        return rows

    def release(self, open_pose):
        self.hold(open_pose, 1.0)
        self._stub.SetTorque(gripper_pb2.TorqueCommand(enable=False))


def sweep(g, start, end, step, dwell, stop_frac):
    """Step from `start` toward `end` in `step`-sized increments along that
    direction (the same posture-preserving extension --grip_assist does),
    dwelling at each. Returns [(travel, cmd, meas_pos, load)]."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    d = end - start
    total = float(np.linalg.norm(d))
    n = d / total
    out = []
    travel = 0.0
    while travel <= total + 1e-9:
        cmd = start + n * travel
        rows = g.hold(cmd, dwell)
        # Settled part of the dwell — but NEVER an empty slice: over a slow link
        # a dwell can yield a single sample (each step costs 2 network round
        # trips), and an empty tail silently poisoned the whole summary with nan.
        tail = rows[len(rows) // 2:] or rows
        meas = (float(np.median([r[0] for r in tail])),
                float(np.median([r[1] for r in tail])))
        load = (float(np.median([r[2] for r in tail])),
                float(np.median([r[3] for r in tail])))
        out.append((travel, tuple(cmd), meas, load))
        peak = max(abs(load[0]), abs(load[1]))
        # Stop once clearly stalled at the cap — the curve past that is flat and
        # there is no reason to keep pressing.
        if stop_frac and peak >= stop_frac * 1000.0 * g._torque:
            print(f"    (stalled at cap: |load| {peak:.0f} — stopping sweep)")
            break
        travel += step
    return out


def main():
    p = argparse.ArgumentParser(description="Gripette load-vs-closure curve")
    p.add_argument("--gripper_addr", default="localhost:50051",
                   help="Real hardware: <pi-ip>:50051. Default is the sim.")
    p.add_argument("--torque", type=float, default=0.25,
                   help="Torque cap (0..1) — use the SAME value the eval will run with")
    p.add_argument("--from", dest="start", type=float, nargs=2, default=[0.0, 0.0],
                   metavar=("PROX", "DIST"), help="Open/start pose")
    p.add_argument("--to", dest="end", type=float, nargs=2, default=[1.2, 1.2],
                   metavar=("PROX", "DIST"),
                   help="Sweep target — the direction defines the posture swept")
    p.add_argument("--step", type=float, default=0.02,
                   help="Increment per step (rad). Match --assist_step.")
    p.add_argument("--dwell", type=float, default=0.15, help="Seconds held per step")
    p.add_argument("--stop_frac", type=float, default=0.95,
                   help="Stop the sweep once |load| reaches this fraction of the cap "
                        "(0 disables)")
    p.add_argument("--out", default="grip_load_curve.jsonl")
    args = p.parse_args()

    g = Gripper(args.gripper_addr, args.torque)
    print(f"connected {args.gripper_addr} | torque cap {args.torque} | "
          f"sweep {args.start} -> {args.end} in {args.step} rad steps")
    print(f"at rest: pos/load {g.read()}")

    trials = {}
    try:
        while True:
            label = input("\nlabel ('air' first, then object name; blank to finish): ").strip()
            if not label:
                break
            g.hold(args.start, 1.0)
            input(f"  place '{label}' in the jaws (clear them for 'air'), Enter to sweep...")
            rows = sweep(g, args.start, args.end, args.step, args.dwell, args.stop_frac)
            trials[label] = rows
            print(f"  {'travel':>7} {'cmd_prox':>9} {'meas_prox':>9} {'load_prox':>9} {'load_dist':>9}")
            for travel, cmd, meas, load in rows:
                print(f"  {travel:7.3f} {cmd[0]:9.3f} {meas[0]:9.3f} "
                      f"{load[0]:9.0f} {load[1]:9.0f}")
            g.hold(args.start, 0.8)   # reopen between trials
            with open(args.out, "a") as f:
                f.write(json.dumps({"label": label, "torque": args.torque,
                                    "step": args.step, "rows": rows}) + "\n")
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted")
    finally:
        g.release(args.start)
        print("gripper reopened, torque released")

    # ---- threshold guidance -------------------------------------------------
    # Aggregate from the LOG, not just this invocation: trials are often run one
    # object per run (place object, sweep, repeat), and the guidance needs the
    # 'air' floor and the objects together to mean anything.
    try:
        for line in open(args.out):
            rec = json.loads(line)
            trials.setdefault(rec["label"], rec["rows"])
    except FileNotFoundError:
        pass
    if not trials:
        return
    print("\n" + "=" * 72)
    print("  LOAD-VS-CLOSURE SUMMARY  (peak |load| per trial)")
    print("=" * 72)
    floor = None
    for label, rows in trials.items():
        peaks = [max(abs(l[0]), abs(l[1])) for _, _, _, l in rows]
        free = float(np.median(peaks[:max(1, len(peaks) // 3)]))   # early = free motion
        print(f"  {label:<10} early/free |load| {free:6.0f} | max |load| {max(peaks):6.0f}")
        if label.lower() == "air":
            floor = max(peaks)
    if floor is not None:
        objs = {k: max(max(abs(l[0]), abs(l[1])) for _, _, _, l in v)
                for k, v in trials.items() if k.lower() != "air"}
        if objs:
            weakest = min(objs.values())
            print(f"\n  air floor (max)      : {floor:.0f}")
            print(f"  weakest object peak  : {weakest:.0f}  ({min(objs, key=objs.get)})")
            if weakest > 2.5 * max(floor, 1.0):
                thr = round((floor + weakest) / 2.0 / 10.0) * 10
                print(f"  --> suggested --grip_assist {thr:.0f}   "
                      f"(comfortably above the floor, reached by every object)")
            else:
                print("  --> MARGIN TOO SMALL: the softest object barely clears the "
                      "air floor.\n      Increase --assist_step (a bigger press gives a "
                      "stronger signal) rather\n      than lowering the threshold into "
                      "the noise, and re-measure.")
    else:
        print("\n  (no 'air' trial — run one: it defines the floor the threshold must clear)")
    print(f"\n  curves logged to {args.out}")


if __name__ == "__main__":
    main()
