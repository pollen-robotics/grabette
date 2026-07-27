"""Bench characterization of the Gripette grabbed-vs-empty signal.

GRIPPER ONLY — no arm, no policy. Decides how the committed-close primitive's
grasp confirmation (evaluate.py --grasp_confirm_load) should work, which sim
could not tell us (present_load is 0 in sim).

The question: after driving a full close at a torque cap, can we tell a real
grasp from a close-on-air? Two candidate signals, and the point of this tool
is to measure which actually separates them on THIS hardware:
  - present_load: rises on contact — BUT a torque-capped empty close may also
    load up against the gripper's own mechanical stop, so load might saturate
    in BOTH cases;
  - final position: a grabbed close stops SHORT at the object's width; an
    empty close reaches (near) full closure — so position may be the cleaner
    discriminator.

Procedure (interactive): for each labelled trial it opens the gripper, waits
for you to place the object (or clear the jaws for an 'air' trial), drives the
close at the torque cap, and logs the load + position trajectory. At the end
it prints a table so you can read off the separation and set the threshold.

Talks to the Gripette gRPC service (real hardware :50051, or the sim — where
load is 0, useful only to check the plumbing). Wire protocol is identical, so
the sim-generated stubs drive the real service.

Usage:
  uv run python examples/characterize_grip_load.py \\
      --gripper_addr <pi-ip>:50051 --torque 0.25 \\
      --closed 1.2 1.0 --open 0.0 0.0 --out grip_char.jsonl
Then, at the prompts, run 'air' plus one trial per object (e.g. mustard, can,
thin). Ctrl-C or 'quit' to finish (torque is released on exit).
"""
import argparse
import json
import time

import grpc
import numpy as np

from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc


def poll(stub, duration_s, hz):
    """Poll ReadMotors for duration_s, returning the recorded trajectory as a
    list of (t_rel, p1, p2, l1, l2)."""
    traj = []
    t0 = time.perf_counter()
    period = 1.0 / hz
    while True:
        t = time.perf_counter() - t0
        if t >= duration_s:
            break
        ms = stub.ReadMotors(gripper_pb2.ReadMotorsRequest())
        traj.append((t, ms.motor1_position, ms.motor2_position,
                     ms.motor1_load, ms.motor2_load))
        time.sleep(period)
    return traj


def send(stub, prox, dist, torque):
    resp = stub.SendMotorCommand(gripper_pb2.MotorCommand(
        motor1_goal=float(prox), motor2_goal=float(dist),
        motor1_torque_limit=float(torque), motor2_torque_limit=float(torque)))
    if not resp.success:
        print(f"  WARNING: SendMotorCommand rejected: {resp.error}")


def summarize(traj):
    """Settled position (median of last 25%) + peak/settled |load| per motor."""
    a = np.array(traj)  # (N,5): t,p1,p2,l1,l2
    tail = a[max(1, int(0.75 * len(a))):]
    return {
        "settled_pos": [float(np.median(tail[:, 1])), float(np.median(tail[:, 2]))],
        "peak_load": [float(np.max(np.abs(a[:, 3]))), float(np.max(np.abs(a[:, 4])))],
        "settled_load": [float(np.median(np.abs(tail[:, 3]))),
                         float(np.median(np.abs(tail[:, 4])))],
    }


def main():
    p = argparse.ArgumentParser(description="Gripette grabbed-vs-empty load/position characterization")
    p.add_argument("--gripper_addr", default="localhost:50051")
    p.add_argument("--torque", type=float, default=0.25,
                   help="Torque cap (fraction 0..1) for the close — match the eval's --grip_torque_limit")
    p.add_argument("--closed", type=float, nargs=2, required=True, metavar=("PROX", "DIST"),
                   help="Fully-closed goal to drive to (your gripper's closed pose)")
    p.add_argument("--open", type=float, nargs=2, default=[0.0, 0.0], metavar=("PROX", "DIST"),
                   help="Open/start pose")
    p.add_argument("--settle_s", type=float, default=2.5,
                   help="Seconds to poll/record after commanding the close")
    p.add_argument("--poll_hz", type=float, default=50.0)
    p.add_argument("--out", default="grip_char.jsonl", help="JSONL trajectory + summary log")
    args = p.parse_args()

    ch = grpc.insecure_channel(args.gripper_addr)
    stub = gripper_pb2_grpc.GripperServiceStub(ch)
    stub.Ping(gripper_pb2.PingRequest())
    print(f"connected to gripper at {args.gripper_addr} | torque cap {args.torque} | "
          f"closed {args.closed} open {args.open}")
    stub.SetTorque(gripper_pb2.TorqueCommand(enable=True))

    trials = []
    try:
        while True:
            label = input("\nlabel for this trial (e.g. air/mustard/can/thin, or 'quit'): ").strip()
            if label.lower() in ("quit", "q", ""):
                break
            # Open and let it settle, so every trial starts from the same pose.
            send(stub, args.open[0], args.open[1], args.torque)
            time.sleep(1.5)
            input(f"  place '{label}' in the jaws (or clear them for an air trial), then Enter to close...")
            send(stub, args.closed[0], args.closed[1], args.torque)
            traj = poll(stub, args.settle_s, args.poll_hz)
            s = summarize(traj)
            trials.append({"label": label, **s})
            print(f"  settled pos {np.round(s['settled_pos'], 4)} | "
                  f"peak |load| {np.round(s['peak_load'], 1)} | "
                  f"settled |load| {np.round(s['settled_load'], 1)}")
            with open(args.out, "a") as f:
                f.write(json.dumps({"label": label, "summary": s, "traj": traj}) + "\n")
    except (KeyboardInterrupt, EOFError):
        print("\ninterrupted")
    finally:
        send(stub, args.open[0], args.open[1], args.torque)  # reopen
        time.sleep(0.5)
        stub.SetTorque(gripper_pb2.TorqueCommand(enable=False))  # release motors
        print("gripper reopened, torque released")

    # --- separation report: grabbed (object) trials vs the 'air' baseline ---
    air = [t for t in trials if t["label"].lower() == "air"]
    obj = [t for t in trials if t["label"].lower() != "air"]
    print("\n" + "=" * 70)
    print("  GRABBED vs EMPTY SEPARATION")
    print("=" * 70)
    hdr = f"  {'trial':<10} {'pos(P,D)':>16} {'peak|load|(P,D)':>18} {'settled|load|':>16}"
    print(hdr)
    for t in trials:
        print(f"  {t['label']:<10} {str(np.round(t['settled_pos'],3)):>16} "
              f"{str(np.round(t['peak_load'],0)):>18} {str(np.round(t['settled_load'],0)):>16}")
    if air and obj:
        ap = np.array([a["settled_pos"] for a in air]).mean(0)
        op = np.array([o["settled_pos"] for o in obj]).mean(0)
        al = np.array([a["settled_load"] for a in air]).mean(0)
        ol = np.array([o["settled_load"] for o in obj]).mean(0)
        print("\n  DISCRIMINATOR:")
        print(f"    position: air closes to {np.round(ap,3)}, objects stop at {np.round(op,3)} "
              f"→ {'SEPARABLE (object stops short)' if np.max(np.abs(ap-op))>0.02 else 'NOT separable'}")
        print(f"    load:     air {np.round(al,0)} vs objects {np.round(ol,0)} "
              f"→ {'SEPARABLE' if np.max(np.abs(al-ol))>5 else 'saturates in both / NOT separable'}")
        print("  → set evaluate.py --grasp_confirm on whichever SEPARATES; if only "
              "position does, the confirm should key on position, not load.")
    else:
        print("\n  (run at least one 'air' trial and one object trial to get the separation)")
    print(f"\n  full trajectories logged to {args.out}")


if __name__ == "__main__":
    main()
