"""Grabette hardware diagnostic — checks all sensors and peripherals.

Run on the Pi 4:
    python scripts/check_hardware.py
"""

import sys
import time


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
    section("IMU (BMI088, I2C bus 3)")
    try:
        from adafruit_extended_bus import ExtendedI2C
        i2c = ExtendedI2C(3)

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

        # Check gyroscope chip ID (should be 0x0F)
        i2c.writeto_then_readfrom(0x69, bytes([0x00]), buf)
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
        # 6G range, 16-bit: 1 LSB = 6*9.81/32768
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


def check_angle_sensors():
    section("Angle Sensors (AS5600)")
    try:
        from adafruit_extended_bus import ExtendedI2C
        results = []

        for bus_num, name in [(4, "Proximal"), (5, "Distal")]:
            try:
                i2c = ExtendedI2C(bus_num)

                # Read angle
                buf = bytearray(2)
                i2c.writeto_then_readfrom(0x36, bytes([0x0C]), buf)
                raw = ((buf[0] & 0x0F) << 8) | buf[1]
                angle = raw * 360.0 / 4096.0

                # Read AGC
                agc_buf = bytearray(1)
                i2c.writeto_then_readfrom(0x36, bytes([0x1A]), agc_buf)
                agc = agc_buf[0]

                # Read status
                status_buf = bytearray(1)
                i2c.writeto_then_readfrom(0x36, bytes([0x0B]), status_buf)
                md = bool(status_buf[0] & 0x08)

                i2c.deinit()

                if md:
                    ok(f"{name} (bus {bus_num}): {angle:.1f}°, AGC={agc}/255, magnet OK")
                else:
                    warn(f"{name} (bus {bus_num}): {angle:.1f}°, AGC={agc}/255, NO MAGNET")
                results.append(True)
            except Exception as e:
                fail(f"{name} (bus {bus_num}): {e}")
                results.append(False)

        return all(results)
    except ImportError:
        fail("adafruit-extended-bus not installed")
        return False


def check_button():
    section("Button (Grove LED, GPIO22/23)")
    try:
        import gpiod
        from gpiod.line import Bias, Direction, Value

        chip_path = None
        for path in ["/dev/gpiochip0", "/dev/gpiochip4"]:
            try:
                gpiod.Chip(path).close()
                chip_path = path
                break
            except Exception:
                continue

        if not chip_path:
            fail("No GPIO chip found")
            return False

        # Test LED
        led_req = gpiod.request_lines(
            chip_path, consumer="test",
            config={22: gpiod.LineSettings(direction=Direction.OUTPUT)},
        )
        led_req.set_value(22, Value.ACTIVE)
        time.sleep(0.2)
        led_req.set_value(22, Value.INACTIVE)
        led_req.release()
        ok(f"LED blinked on {chip_path} pin 22")

        # Test button read
        btn_req = gpiod.request_lines(
            chip_path, consumer="test",
            config={23: gpiod.LineSettings(
                direction=Direction.INPUT, bias=Bias.PULL_UP,
            )},
        )
        val = btn_req.get_value(23)
        btn_req.release()
        pressed = (val == Value.INACTIVE)
        ok(f"Button state: {'PRESSED' if pressed else 'released'}")

        return True
    except ImportError:
        fail("gpiod not installed")
        return False
    except Exception as e:
        fail(str(e))
        return False


def check_bluetooth():
    section("Bluetooth Service")
    try:
        import subprocess
        result = subprocess.run(
            ["systemctl", "is-active", "grabette-bluetooth"],
            capture_output=True, text=True,
        )
        status = result.stdout.strip()
        if status == "active":
            ok("grabette-bluetooth.service is running")
            return True
        else:
            warn(f"grabette-bluetooth.service is {status}")
            return False
    except Exception as e:
        fail(str(e))
        return False


def main():
    print("Grabette Hardware Diagnostic")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    results["Camera"] = check_camera()
    results["IMU"] = check_imu()
    results["Angle Sensors"] = check_angle_sensors()
    results["Button"] = check_button()
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
