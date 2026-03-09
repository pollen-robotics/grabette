"""Calibrate AS5600 angle sensor offsets.

Place both fingers at the zero/rest position, then run this script.
It reads the raw angles and saves them as offsets so that the zero
position reads 0 rad after calibration.

Must run on the Pi (needs I2C access).

Usage:
    python scripts/calibrate_angles.py          # read & save
    python scripts/calibrate_angles.py --read   # just read raw values, don't save
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

CALIBRATION_FILE = Path.home() / ".grabette" / "angle_calibration.json"
AS5600_ADDRESS = 0x36
ANGLE_REGISTER = 0x0C
I2C_BUS_1 = 4  # sensor 1 (distal)
I2C_BUS_2 = 5  # sensor 2 (proximal)
NUM_SAMPLES = 20  # average over N reads for stability


def read_raw_angle(i2c) -> float:
    """Read raw angle in degrees from AS5600."""
    result = bytearray(2)
    i2c.writeto_then_readfrom(AS5600_ADDRESS, bytes([ANGLE_REGISTER]), result)
    raw = ((result[0] & 0x0F) << 8) | result[1]
    return raw * 360.0 / 4096.0


def read_averaged(i2c, n: int) -> float:
    """Read N samples and return the average (handles wraparound at 360/0)."""
    import math
    # Use circular mean to handle wraparound
    sin_sum = 0.0
    cos_sum = 0.0
    for _ in range(n):
        deg = read_raw_angle(i2c)
        rad = math.radians(deg)
        sin_sum += math.sin(rad)
        cos_sum += math.cos(rad)
        time.sleep(0.01)
    avg_rad = math.atan2(sin_sum / n, cos_sum / n)
    return math.degrees(avg_rad) % 360.0


def main():
    parser = argparse.ArgumentParser(description="Calibrate AS5600 angle sensor offsets")
    parser.add_argument("--read", action="store_true", help="Just read raw values, don't save")
    args = parser.parse_args()

    from adafruit_extended_bus import ExtendedI2C

    i2c_1 = ExtendedI2C(I2C_BUS_1)
    i2c_2 = ExtendedI2C(I2C_BUS_2)

    print(f"Reading {NUM_SAMPLES} samples from each sensor...")
    raw1 = read_averaged(i2c_1, NUM_SAMPLES)
    raw2 = read_averaged(i2c_2, NUM_SAMPLES)

    i2c_1.deinit()
    i2c_2.deinit()

    print(f"\nRaw angles at current position:")
    print(f"  Sensor 1 (distal,   bus {I2C_BUS_1}): {raw1:.1f}°")
    print(f"  Sensor 2 (proximal, bus {I2C_BUS_2}): {raw2:.1f}°")

    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE) as f:
            old = json.load(f)
        print(f"\nCurrent calibration:")
        print(f"  Sensor 1 offset: {old.get('sensor_1_offset_deg', 0):.1f}°")
        print(f"  Sensor 2 offset: {old.get('sensor_2_offset_deg', 0):.1f}°")

    if args.read:
        return

    # Save new calibration
    calibration = {
        "sensor_1_offset_deg": round(raw1, 6),
        "sensor_2_offset_deg": round(raw2, 6),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nNew calibration saved to {CALIBRATION_FILE}:")
    print(f"  Sensor 1 offset: {raw1:.1f}°")
    print(f"  Sensor 2 offset: {raw2:.1f}°")
    print("\nRestart grabette for changes to take effect.")


if __name__ == "__main__":
    main()
