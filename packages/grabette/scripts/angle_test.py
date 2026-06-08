"""Read proximal angle sensor (AS5600) and its AGC register.

Run on the Pi:
    python scripts/angle_test.py
"""

import time

from adafruit_extended_bus import ExtendedI2C

AS5600_ADDR = 0x36
I2C_BUS_DISTAL = 5  # distal sensor on bus 5

# AS5600 registers
REG_ANGLE = 0x0C    # 12-bit angle (2 bytes)
REG_AGC = 0x1A      # Automatic Gain Control (1 byte)
REG_STATUS = 0x0B   # Magnet status (1 byte)


def read_angle(i2c) -> float:
    """Read raw angle in degrees (0-360)."""
    buf = bytearray(2)
    i2c.writeto_then_readfrom(AS5600_ADDR, bytes([REG_ANGLE]), buf)
    raw = ((buf[0] & 0x0F) << 8) | buf[1]
    return raw * 360.0 / 4096.0


def read_agc(i2c) -> int:
    """Read AGC value (0-255). ~128 is ideal; 0=magnet too strong, 255=too weak."""
    buf = bytearray(1)
    i2c.writeto_then_readfrom(AS5600_ADDR, bytes([REG_AGC]), buf)
    return buf[0]


def read_status(i2c) -> dict:
    """Read magnet status register."""
    buf = bytearray(1)
    i2c.writeto_then_readfrom(AS5600_ADDR, bytes([REG_STATUS]), buf)
    val = buf[0]
    return {
        "raw": val,
        "magnet_detected": bool(val & 0x08),   # bit 3: MD
        "magnet_too_weak": bool(val & 0x10),    # bit 4: ML
        "magnet_too_strong": bool(val & 0x20),  # bit 5: MH
    }


def main():
    i2c = ExtendedI2C(I2C_BUS_DISTAL)
    print(f"AS5600 on I2C bus {I2C_BUS_DISTAL}, addr 0x{AS5600_ADDR:02X}")

    status = read_status(i2c)
    print(f"Status: detected={status['magnet_detected']}, "
          f"too_weak={status['magnet_too_weak']}, "
          f"too_strong={status['magnet_too_strong']}")
    print()

    try:
        while True:
            angle = read_angle(i2c)
            agc = read_agc(i2c)
            status = read_status(i2c)
            md = "MD" if status["magnet_detected"] else "  "
            ml = "ML" if status["magnet_too_weak"] else "  "
            mh = "MH" if status["magnet_too_strong"] else "  "
            print(f"\r  Angle: {angle:6.1f}°   AGC: {agc:3d}/255   Status: 0x{status['raw']:02X} [{md} {ml} {mh}]", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDone")
    finally:
        i2c.deinit()


if __name__ == "__main__":
    main()
