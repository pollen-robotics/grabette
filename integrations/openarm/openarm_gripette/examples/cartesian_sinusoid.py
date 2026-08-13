"""Send a clean Cartesian sinusoid via SendCartesianDelta to diagnose jerkiness.

Purpose: if the arm moves jerkily when the policy drives it, is the command
stream noisy, or is the server-side IK/motor stack introducing the jerk? This
script removes the policy from the equation and sends a mathematically smooth
sinusoid on one axis. If that still looks jerky on the arm, the issue is
downstream (IK jacobian weirdness, stiff MIT gains, motor tracking).

Defaults are conservative: 2 cm amplitude, 0.25 Hz (4 s period), on the Y axis
(side-to-side — least gravity interaction). Ramps in from zero over 2 s and
ramps back out at the end. A hard safety cap on per-step displacement aborts
before sending if the chosen amplitude/frequency would produce a step larger
than MAX_PER_STEP_MM.

Usage:
  # Safe defaults (Y-axis sinusoid, ±2 cm at 0.25 Hz, 10 s run)
  uv run python examples/cartesian_sinusoid.py \\
      --arm_addr 192.168.10.147:50052

  # Faster / larger sweep (still sanity-checked against MAX_PER_STEP_MM)
  uv run python examples/cartesian_sinusoid.py \\
      --arm_addr 192.168.10.147:50052 --amplitude_mm 30 --frequency_hz 0.5

  # Test a different axis
  uv run python examples/cartesian_sinusoid.py \\
      --arm_addr 192.168.10.147:50052 --axis z
"""

import argparse
import logging
import time

import grpc
import numpy as np
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc

from _torque_guard import abort_torque_off, add_keep_torque_arg

logger = logging.getLogger(__name__)

# Hard cap on per-step displacement. If the target sinusoid's peak per-step
# delta exceeds this, the script aborts before sending anything. This is the
# single line of defence against a user asking for a dangerously fast motion.
MAX_PER_STEP_MM = 4.0

# Ramp-in/out duration (seconds). Amplitude is multiplied by a 0->1->0 window
# so the motion starts and ends at zero velocity.
DEFAULT_RAMP_S = 2.0


def parse_args():
    p = argparse.ArgumentParser(description="Cartesian sinusoid diagnostic")
    p.add_argument("--arm_addr", type=str, default="localhost:50052")
    p.add_argument("--axis", type=str, default="y", choices=["x", "y", "z"], help="Axis to oscillate on")
    p.add_argument("--amplitude_mm", type=float, default=20.0, help="Peak amplitude from center (mm)")
    p.add_argument("--frequency_hz", type=float, default=0.25, help="Sinusoid frequency (Hz)")
    p.add_argument("--fps", type=float, default=10.0, help="Command rate (matches typical policy rate)")
    p.add_argument("--duration", type=float, default=10.0, help="Total run duration (s)")
    p.add_argument(
        "--ramp_s",
        type=float,
        default=DEFAULT_RAMP_S,
        help="Ramp-in/out duration (s). Amplitude grows 0->A over ramp_s at start, A->0 at end.",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV file to dump per-step commanded + actual pose for plotting.",
    )
    add_keep_torque_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

    axis_index = {"x": 0, "y": 1, "z": 2}[args.axis]
    dt = 1.0 / args.fps
    omega = 2.0 * np.pi * args.frequency_hz
    amplitude_m = args.amplitude_mm / 1000.0

    # Peak per-step velocity estimate: A * omega. Peak per-step displacement = A * omega * dt.
    peak_step_mm = amplitude_m * omega * dt * 1000.0
    logger.info(
        f"Sinusoid: axis={args.axis}, A={args.amplitude_mm:.1f}mm, f={args.frequency_hz}Hz, "
        f"fps={args.fps}Hz -> peak per-step = {peak_step_mm:.2f}mm"
    )
    if peak_step_mm > MAX_PER_STEP_MM:
        raise SystemExit(
            f"Peak per-step delta {peak_step_mm:.2f}mm exceeds safety cap {MAX_PER_STEP_MM}mm. "
            "Lower --amplitude_mm or --frequency_hz."
        )

    # ---- Connect and capture start pose ----
    channel = grpc.insecure_channel(args.arm_addr)
    stub = arm_pb2_grpc.ArmServiceStub(channel)
    stub.Ping(arm_pb2.ArmPingRequest())
    logger.info(f"Connected to {args.arm_addr}")

    state = stub.GetArmState(arm_pb2.GetArmStateRequest())
    start_pos = np.array([state.x, state.y, state.z], dtype=np.float64)
    logger.info(f"Start pos: {start_pos.round(4).tolist()}")

    # commanded_offset[axis] tracks the integrated commanded displacement from start.
    # Each SendCartesianDelta accumulates on the server target, so we send the *difference*
    # between the target value at this step and the previous step.
    prev_commanded = 0.0

    csv_file = open(args.csv, "w") if args.csv else None
    if csv_file:
        csv_file.write("t,commanded_offset,actual_offset,raw_delta_mm\n")

    logger.info(f"Running for {args.duration}s at {args.fps}Hz (Ctrl+C to abort)")
    t0 = time.perf_counter()
    step = 0
    try:
        while True:
            tick = time.perf_counter()
            t = tick - t0
            if t >= args.duration:
                break

            # Ramp: 0 during ramp-in, 1 while steady, 0 during ramp-out.
            ramp_in = min(1.0, t / args.ramp_s) if args.ramp_s > 0 else 1.0
            time_remaining = args.duration - t
            ramp_out = min(1.0, time_remaining / args.ramp_s) if args.ramp_s > 0 else 1.0
            envelope = min(ramp_in, ramp_out)

            # Commanded target offset from start on the chosen axis.
            target = envelope * amplitude_m * np.sin(omega * t)

            # Per-step delta is the change since the previous commanded value.
            delta_axis = target - prev_commanded
            prev_commanded = target

            delta = np.zeros(3)
            delta[axis_index] = delta_axis
            dr6d = [0.0] * 6  # no rotation change — purely translational sinusoid

            stub.SendCartesianDelta(
                arm_pb2.CartesianDelta(
                    dx=float(delta[0]),
                    dy=float(delta[1]),
                    dz=float(delta[2]),
                    dr6d=dr6d,
                )
            )

            # Read back actual pose for comparison.
            state = stub.GetArmState(arm_pb2.GetArmStateRequest())
            actual_pos = np.array([state.x, state.y, state.z], dtype=np.float64)
            actual_offset = actual_pos[axis_index] - start_pos[axis_index]

            delta_mm = abs(delta_axis) * 1000.0
            if step % 5 == 0:
                logger.info(
                    f"t={t:5.2f}s step={step:4d} | "
                    f"commanded {target * 1000:+7.2f}mm | actual {actual_offset * 1000:+7.2f}mm | "
                    f"delta {delta_mm:5.2f}mm | envelope {envelope:.2f}"
                )

            if csv_file:
                csv_file.write(f"{t:.4f},{target:.6f},{actual_offset:.6f},{delta_mm:.4f}\n")

            step += 1

            elapsed = time.perf_counter() - tick
            sleep_for = dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
        abort_torque_off(stub, args.keep_torque)
    except Exception:
        abort_torque_off(stub, args.keep_torque)
        raise
    finally:
        if csv_file:
            csv_file.close()
            logger.info(f"CSV written to {args.csv}")
        channel.close()
        logger.info("Done.")


if __name__ == "__main__":
    main()
