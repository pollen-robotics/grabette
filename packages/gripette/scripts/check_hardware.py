"""Gripette hardware diagnostic — checks all sensors and peripherals.

Run on the Pi Zero 2W:
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
        ok(f"1296x972, exposure={exposure}us, gain={gain:.1f}")
        return True
    except ImportError:
        fail("picamera2 not installed")
        return False
    except Exception as e:
        fail(str(e))
        return False


def check_motors():
    section("Motors (STS3215, /dev/ttyS0)")
    try:
        from gripette.config import settings
        import serial

        # Check serial port exists and is accessible
        try:
            ser = serial.Serial(settings.motor_port, settings.motor_baudrate, timeout=0.1)
            discarded = ser.read(4096)
            ser.close()
            if discarded:
                warn(f"Flushed {len(discarded)} stale bytes from {settings.motor_port}")
            ok(f"Serial port {settings.motor_port} @ {settings.motor_baudrate} baud")
        except Exception as e:
            fail(f"Serial port {settings.motor_port}: {e}")
            return False

        # Check motors via rustypot
        try:
            from rustypot import Sts3215PyController
        except ImportError:
            fail("rustypot not installed")
            return False

        ctrl = Sts3215PyController(settings.motor_port, settings.motor_baudrate, 1.0)
        ids = [settings.motor_id_1, settings.motor_id_2]
        try:
            pos = ctrl.sync_read_present_position(ids)
            import math
            ok(f"Motor {ids[0]}: {math.degrees(pos[0]):.1f}° ({pos[0]:.3f} rad)")
            ok(f"Motor {ids[1]}: {math.degrees(pos[1]):.1f}° ({pos[1]:.3f} rad)")
            return True
        except RuntimeError as e:
            fail(f"Communication error: {e}")
            return False
    except Exception as e:
        fail(str(e))
        return False


def check_bluetooth():
    section("Bluetooth Service")
    try:
        import subprocess
        result = subprocess.run(
            ["systemctl", "is-active", "gripette-bluetooth"],
            capture_output=True, text=True,
        )
        status = result.stdout.strip()
        if status == "active":
            ok("gripette-bluetooth.service is running")
            return True
        else:
            warn(f"gripette-bluetooth.service is {status}")
            return False
    except Exception as e:
        fail(str(e))
        return False


def check_grpc_service():
    section("gRPC Service")
    try:
        import subprocess
        result = subprocess.run(
            ["systemctl", "is-active", "gripette"],
            capture_output=True, text=True,
        )
        status = result.stdout.strip()
        if status == "active":
            ok("gripette.service is running")
            return True
        else:
            warn(f"gripette.service is {status}")
            return False
    except Exception as e:
        fail(str(e))
        return False


def main():
    print("Gripette Hardware Diagnostic")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    results["Camera"] = check_camera()
    results["Motors"] = check_motors()
    results["Bluetooth"] = check_bluetooth()
    results["gRPC Service"] = check_grpc_service()

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
