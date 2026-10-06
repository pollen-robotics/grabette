"""Calibrate AS5600 angle sensor offsets.

Place both fingers at the FULLY OPEN position (the convention's zero), then
run this script. It reads the raw angles and saves them as offsets so that
this pose reads 0 rad after calibration. The runtime then applies the per-
sensor sign (derived from settings.hand) to produce the robot-frame output
convention: 0 = fully open, positive = closing.

Sign flipping is independent of this calibration — it happens downstream in
hardware/angle.py at sample time. Recalibrating after changing `hand` is NOT
required.

Must run on the Pi (needs I2C access). The dashboard does the same from its
"Calibrate my device" button, without the restart.

Usage:
    python scripts/calibrate_angles.py          # read & save
    python scripts/calibrate_angles.py --read   # just read raw values, don't save
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grabette.hardware.angle import (
    CALIBRATION_FILE,
    CALIBRATION_SAMPLES,
    AngleCapture,
    load_calibration,
    read_raw_averaged,
    save_calibration,
)

AS5600_ADDRESS = AngleCapture.AS5600_ADDRESS
ANGLE_REGISTER = AngleCapture.ANGLE_REGISTER
I2C_BUS_1 = 3  # sensor 1 (distal,   V2 = /dev/i2c-3)
I2C_BUS_2 = 4  # sensor 2 (proximal, V2 = /dev/i2c-4)
NUM_SAMPLES = CALIBRATION_SAMPLES  # average over N reads for stability


def read_raw_angle(i2c) -> float:
    """Read raw angle in degrees from AS5600."""
    result = bytearray(2)
    i2c.writeto_then_readfrom(AS5600_ADDRESS, bytes([ANGLE_REGISTER]), result)
    raw = ((result[0] & 0x0F) << 8) | result[1]
    return raw * 360.0 / 4096.0


def main():
    parser = argparse.ArgumentParser(description="Calibrate AS5600 angle sensor offsets")
    parser.add_argument("--read", action="store_true", help="Just read raw values, don't save")
    args = parser.parse_args()

    from adafruit_extended_bus import ExtendedI2C

    i2c_1 = ExtendedI2C(I2C_BUS_1)
    i2c_2 = ExtendedI2C(I2C_BUS_2)

    print(f"Reading {NUM_SAMPLES} samples from each sensor...")
    raw1 = read_raw_averaged(lambda: read_raw_angle(i2c_1), NUM_SAMPLES)
    raw2 = read_raw_averaged(lambda: read_raw_angle(i2c_2), NUM_SAMPLES)

    i2c_1.deinit()
    i2c_2.deinit()

    print("\nRaw angles at current position:")
    print(f"  Sensor 1 (distal,   bus {I2C_BUS_1}): {raw1:.1f}°")
    print(f"  Sensor 2 (proximal, bus {I2C_BUS_2}): {raw2:.1f}°")

    old = load_calibration()
    if old is not None:
        print("\nCurrent calibration:")
        print(f"  Sensor 1 offset: {old.get('sensor_1_offset_deg', 0):.1f}°")
        print(f"  Sensor 2 offset: {old.get('sensor_2_offset_deg', 0):.1f}°")

    if args.read:
        return

    save_calibration(raw1, raw2)

    print(f"\nNew calibration saved to {CALIBRATION_FILE}:")
    print(f"  Sensor 1 offset: {raw1:.1f}°")
    print(f"  Sensor 2 offset: {raw2:.1f}°")
    print("\nRestart grabette for changes to take effect.")


if __name__ == "__main__":
    main()
