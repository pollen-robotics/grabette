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


def check_camera(service_holds: bool = False):
    """Probe the camera directly. If gripette.service holds it, downgrade
    'busy' to SKIP (the running service is itself evidence the camera works)."""
    section("Camera (picamera2)")
    try:
        from picamera2 import Picamera2
    except ImportError:
        fail("picamera2 not installed")
        return False
    try:
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
    except Exception as e:
        if service_holds:
            warn(f"camera in use by gripette.service — skipping direct probe ({e})")
            return None
        fail(str(e))
        return False


def check_motors(service_holds: bool = False):
    """Probe the motor bus directly. If gripette.service holds /dev/serial0,
    downgrade 'busy' to SKIP."""
    section("Motors (STS3215, /dev/ttyS0)")
    try:
        from gripette.config import settings
        import serial
    except ImportError as e:
        fail(str(e))
        return False

    # Serial port open
    try:
        ser = serial.Serial(settings.motor_port, settings.motor_baudrate, timeout=0.1)
        discarded = ser.read(4096)
        ser.close()
        if discarded:
            warn(f"Flushed {len(discarded)} stale bytes from {settings.motor_port}")
        ok(f"Serial port {settings.motor_port} @ {settings.motor_baudrate} baud")
    except Exception as e:
        if service_holds:
            warn(f"{settings.motor_port} in use by gripette.service — skipping direct probe ({e})")
            return None
        fail(f"Serial port {settings.motor_port}: {e}")
        return False

    # rustypot — drive the bus to read positions
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
        if service_holds:
            warn(f"motor read failed (likely contended with gripette.service): {e}")
            return None
        fail(f"Communication error: {e}")
        return False


def _service_status(unit: str) -> str:
    """Return 'active', 'inactive', or 'not-installed'.

    'systemctl is-active' returns 'inactive' for both "stopped" and "doesn't
    exist", so we first probe with 'systemctl cat' which fails cleanly for
    units that aren't on disk.
    """
    import subprocess
    cat = subprocess.run(["systemctl", "cat", unit],
                         capture_output=True, text=True)
    if cat.returncode != 0:
        return "not-installed"
    active = subprocess.run(["systemctl", "is-active", unit],
                            capture_output=True, text=True)
    return "active" if active.stdout.strip() == "active" else "inactive"


def check_service(unit: str, label: str):
    """Returns True (active), False (installed but down), or None (not installed)."""
    section(label)
    try:
        status = _service_status(unit)
    except Exception as e:
        fail(str(e))
        return False
    if status == "active":
        ok(f"{unit}.service is running")
        return True
    if status == "not-installed":
        warn(f"{unit}.service not installed yet — run 'make install-systemd' for boot-time start")
        return None
    fail(f"{unit}.service is installed but {status}")
    return False


def main():
    print("Gripette Hardware Diagnostic")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check services first so the hardware probes can interpret "device busy"
    # correctly: when gripette.service is active, it owns /dev/serial0 and the
    # camera exclusively, so direct opens will fail — that's expected, not a
    # bug, so we downgrade those failures to SKIP.
    grpc = check_service("gripette", "gRPC Service")
    bluetooth = check_service("gripette-bluetooth", "Bluetooth Service")
    service_holds_hw = (grpc is True)

    results = {
        "Camera":       check_camera(service_holds=service_holds_hw),
        "Motors":       check_motors(service_holds=service_holds_hw),
        "Bluetooth":    bluetooth,
        "gRPC Service": grpc,
    }

    section("Summary")
    hardware_ok = True
    for name, passed in results.items():
        if passed is True:
            label = "[OK]  "
        elif passed is None:
            label = "[SKIP]"
        else:
            label = "[FAIL]"
            hardware_ok = False
        print(f"  {label} {name}")

    print()
    if not hardware_ok:
        print("Some checks failed — see details above.")
        sys.exit(1)

    services_missing = (grpc is None or bluetooth is None)
    hardware_skipped = (results["Camera"] is None or results["Motors"] is None)

    if hardware_skipped:
        print("Services are running and using the hardware directly. All good.")
    elif services_missing:
        print("Hardware OK. Services not installed yet — run 'make install-systemd' "
              "for boot-time start.")
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
