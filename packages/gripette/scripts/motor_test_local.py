"""Local motor test — talks directly to hardware, no gRPC.

Run on the Pi to verify serial communication and motor movement.

Usage:
    python scripts/motor_test_local.py
"""

import math
import time

from gripette.config import settings
from gripette.hardware.motors import MotorController

FREQ = 1.0       # Hz
DURATION = 3.0   # seconds
LOOP_HZ = 50

# Motor 1: center of range, amplitude 0.3 rad
M1_CENTER = (settings.motor1_min + settings.motor1_max) / 2
M1_AMP = 0.3
# Motor 2: center of range, amplitude 0.3 rad
M2_CENTER = (settings.motor2_min + settings.motor2_max) / 2
M2_AMP = 0.3


def main():
    motors = MotorController(
        port=settings.motor_port,
        baudrate=settings.motor_baudrate,
        id_1=settings.motor_id_1,
        id_2=settings.motor_id_2,
        limits=(
            (settings.motor1_min, settings.motor1_max),
            (settings.motor2_min, settings.motor2_max),
        ),
    )

    print(f"Starting motors on {settings.motor_port}...")
    motors.start()

    pos = motors.read_positions()
    print(f"Current positions: ({pos[0]:.3f}, {pos[1]:.3f}) rad")

    print(f"Torque on — sinus {FREQ}Hz for {DURATION}s")
    motors.set_torque(True)

    dt = 1.0 / LOOP_HZ
    t0 = time.monotonic()
    next_time = t0

    try:
        while True:
            t = time.monotonic() - t0
            if t > DURATION:
                break

            phase = 2 * math.pi * FREQ * t
            cmd1 = M1_CENTER + M1_AMP * math.sin(phase)
            cmd2 = M2_CENTER + M2_AMP * math.sin(phase)

            motors.write_goal_positions(cmd1, cmd2)
            fb1, fb2 = motors.read_positions()
            print(f"\r  t={t:.2f}s  cmd=({cmd1:.3f}, {cmd2:.3f})  fb=({fb1:.3f}, {fb2:.3f})", end="")

            next_time += dt
            sleep_dur = next_time - time.monotonic()
            if sleep_dur > 0:
                time.sleep(sleep_dur)
    finally:
        print()
        motors.set_torque(False)
        print("Torque off")
        motors.stop()
        print("Done")


if __name__ == "__main__":
    main()
