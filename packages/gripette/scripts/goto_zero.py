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

TARGET = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.36:50051"


def main():

    with GripperClient(TARGET) as g:
        print(f"Connected to {TARGET}")
        g.torque_on()


        # Send command + read feedback
        g.move(0.0, 0.0)
        time.sleep(1.0)
        fb1, fb2 = g.read_motors()
        print(f'Positions: {fb1} {fb2}')
        g.torque_off()
        print("Torque off")


if __name__ == "__main__":
    main()
