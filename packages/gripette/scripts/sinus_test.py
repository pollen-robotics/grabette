"""Send sinusoidal commands to motors and record feedback for delay analysis.

Sends commands and reads feedback at LOOP_HZ using the lightweight ReadMotors RPC
(no camera involved). Outputs a CSV for plotting command vs feedback.

Usage:
    uv run python scripts/sinus_test.py [host:port]

Plot:
    import pandas as pd, matplotlib.pyplot as plt
    df = pd.read_csv("sinus_test.csv")
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(df.timestamp_s, df.cmd1, label="cmd"); ax1.plot(df.timestamp_s, df.fb1, label="fb")
    ax1.set_ylabel("Motor 1 (rad)"); ax1.legend()
    ax2.plot(df.timestamp_s, df.cmd2, label="cmd"); ax2.plot(df.timestamp_s, df.fb2, label="fb")
    ax2.set_ylabel("Motor 2 (rad)"); ax2.legend(); ax2.set_xlabel("Time (s)")
    plt.tight_layout(); plt.savefig("sinus_test.png", dpi=150); plt.show()
"""

import csv
import math
import sys
import time

from gripette.client import GripperClient

# Sinus parameters
FREQ = 1.0         # Hz
DURATION = 5.0     # seconds
LOOP_HZ = 100      # command + feedback rate

# Motor 1: center of range [-1.484, 0] → -0.742, amplitude 0.3 rad
M1_CENTER = -0.742
M1_AMP = 0.3
# Motor 2: center of range [-2.025, 0] → -1.013, amplitude 0.3 rad
M2_CENTER = -1.013
M2_AMP = 0.3

OUTPUT = "sinus_test.csv"
TARGET = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.36:50051"


def main():
    rows = []
    dt = 1.0 / LOOP_HZ

    with GripperClient(TARGET) as g:
        print(f"Connected to {TARGET}")
        g.torque_on()
        print(f"Torque on — sinus test: {FREQ}Hz, {DURATION}s, loop at {LOOP_HZ}Hz")

        t0 = time.monotonic()
        next_time = t0

        while True:
            t = time.monotonic() - t0
            if t > DURATION:
                break

            # Compute sinusoidal commands
            phase = 2 * math.pi * FREQ * t
            cmd1 = M1_CENTER + M1_AMP * math.sin(phase)
            cmd2 = M2_CENTER + M2_AMP * math.sin(phase)

            # Send command + read feedback
            g.move(cmd1, cmd2)
            fb1, fb2 = g.read_motors()

            rows.append({
                "timestamp_s": round(t, 4),
                "cmd1": round(cmd1, 4),
                "cmd2": round(cmd2, 4),
                "fb1": round(fb1, 4),
                "fb2": round(fb2, 4),
            })

            # Rate limiting
            next_time += dt
            sleep_dur = next_time - time.monotonic()
            if sleep_dur > 0:
                time.sleep(sleep_dur)

        g.torque_off()
        print("Torque off")

    # Write CSV
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_s", "cmd1", "cmd2", "fb1", "fb2"])
        writer.writeheader()
        writer.writerows(rows)

    actual_hz = len(rows) / DURATION
    print(f"Wrote {len(rows)} samples to {OUTPUT} ({actual_hz:.0f}Hz effective)")


if __name__ == "__main__":
    main()
