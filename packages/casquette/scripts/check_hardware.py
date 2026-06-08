"""Casquette hardware diagnostic — checks camera and IMU.

Run on the Pi:
    uv run python scripts/check_hardware.py
"""

import sys
import time

from casquette.config import settings


def section(name):
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")


def ok(msg):
    print(f"  [OK]   {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def check_camera():
    section("Camera (picamera2)")
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_still_configuration(main={"size": (1296, 972)})
        cam.configure(config)
        cam.start()
        time.sleep(0.5)
        metadata = cam.capture_metadata()
        cam.stop()
        cam.close()

        exposure = metadata.get("ExposureTime", "?")
        gain = metadata.get("AnalogueGain", "?")
        sensor_ts = metadata.get("SensorTimestamp")
        ok(f"1296x972, exposure={exposure}us, gain={gain:.1f}")
        if sensor_ts:
            ok(f"SensorTimestamp available ({sensor_ts})")
        else:
            warn("SensorTimestamp not available")
        return True
    except ImportError:
        fail("picamera2 not installed")
        return False
    except Exception as e:
        fail(str(e))
        return False


def check_imu():
    bus = settings.imu_i2c_bus
    section(f"IMU (BMI088, I2C bus {bus})")
    try:
        from adafruit_extended_bus import ExtendedI2C
        i2c = ExtendedI2C(bus)

        # Check accelerometer chip ID (should be 0x1E)
        buf = bytearray(1)
        i2c.writeto_then_readfrom(0x19, bytes([0x00]), buf)
        # BMI088 accel needs a dummy read first
        i2c.writeto_then_readfrom(0x19, bytes([0x00]), buf)
        accel_id = buf[0]
        if accel_id == 0x1E:
            ok(f"Accelerometer chip ID: 0x{accel_id:02X}")
        else:
            fail(f"Accelerometer chip ID: 0x{accel_id:02X} (expected 0x1E)")

        # Check gyroscope chip ID (should be 0x0F). HAT BMI088 has SDO2
        # strapped low, putting the gyro at 0x68 (the BMI088 driver's
        # DEFAULT_GYRO_ADDR matches).
        from casquette.hardware.bmi088 import DEFAULT_GYRO_ADDR
        i2c.writeto_then_readfrom(DEFAULT_GYRO_ADDR, bytes([0x00]), buf)
        gyro_id = buf[0]
        if gyro_id == 0x0F:
            ok(f"Gyroscope chip ID: 0x{gyro_id:02X}")
        else:
            fail(f"Gyroscope chip ID: 0x{gyro_id:02X} (expected 0x0F)")

        # Read raw accel data
        data = bytearray(6)
        i2c.writeto_then_readfrom(0x19, bytes([0x12]), data)
        ax = int.from_bytes(data[0:2], "little", signed=True)
        ay = int.from_bytes(data[2:4], "little", signed=True)
        az = int.from_bytes(data[4:6], "little", signed=True)
        scale = 6 * 9.81 / 32768
        ok(f"Accel raw: ({ax*scale:.2f}, {ay*scale:.2f}, {az*scale:.2f}) m/s²")

        i2c.deinit()
        return True
    except ImportError:
        fail("adafruit-extended-bus not installed")
        return False
    except Exception as e:
        fail(str(e))
        return False


def check_bluetooth():
    section("Bluetooth Service")
    try:
        import subprocess
        result = subprocess.run(
            ["systemctl", "is-active", "casquette-bluetooth"],
            capture_output=True, text=True,
        )
        status = result.stdout.strip()
        if status == "active":
            ok("casquette-bluetooth.service is running")
            return True
        else:
            warn(f"casquette-bluetooth.service is {status}")
            return False
    except Exception as e:
        fail(str(e))
        return False


def main():
    print("Casquette Hardware Diagnostic")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    results["Camera"] = check_camera()
    results["IMU"] = check_imu()
    results["Bluetooth"] = check_bluetooth()

    section("Summary")
    all_ok = True
    for name, passed in results.items():
        status = "[OK]  " if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print("All checks passed.")
    else:
        print("Some checks failed — see details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
