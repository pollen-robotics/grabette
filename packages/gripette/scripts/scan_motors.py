"""Scan a serial bus for Feetech STS3215 servos.

Pings each ID in [--start, --end] and reports which ones reply. Useful when
a motor has been reconfigured to an unknown ID, or when checking that both
servos are responding on the bus.

Usage:
    python scripts/scan_motors.py
    python scripts/scan_motors.py --port /dev/ttyUSB0 --baudrate 1000000
    python scripts/scan_motors.py --start 1 --end 10
"""

import argparse
import math
import sys
import time

import serial

from gripette.config import settings

try:
    from rustypot import Sts3215PyController
except ImportError:
    print("rustypot not installed — run `make install-rpi` first.", file=sys.stderr)
    sys.exit(1)

# Short per-ID timeout: STS3215 replies in <2 ms at 1 Mbps, so 30 ms is plenty
# for a real motor and keeps a full 1..253 scan under ~8 s of dead air.
SCAN_TIMEOUT = 0.03


def flush_serial(port: str, baudrate: int) -> None:
    """Drain stale bytes (e.g. Pi boot console on /dev/ttyS0)."""
    try:
        ser = serial.Serial(port, baudrate, timeout=0.1)
        discarded = ser.read(4096)
        ser.close()
        if discarded:
            print(f"Flushed {len(discarded)} stale bytes from {port}")
    except Exception as e:
        print(f"Warning: could not flush {port}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default=settings.motor_port,
                        help=f"Serial port (default: {settings.motor_port})")
    parser.add_argument("--baudrate", type=int, default=settings.motor_baudrate,
                        help=f"Baudrate (default: {settings.motor_baudrate})")
    parser.add_argument("--start", type=int, default=1,
                        help="First ID to scan (default: 1)")
    parser.add_argument("--end", type=int, default=253,
                        help="Last ID to scan, inclusive (default: 253)")
    args = parser.parse_args()

    if not (0 <= args.start <= args.end <= 253):
        parser.error("IDs must satisfy 0 <= --start <= --end <= 253")

    print(f"Scanning {args.port} @ {args.baudrate} baud, IDs {args.start}..{args.end}")
    flush_serial(args.port, args.baudrate)

    ctrl = Sts3215PyController(args.port, args.baudrate, SCAN_TIMEOUT)

    found: list[tuple[int, float]] = []
    t0 = time.monotonic()
    for motor_id in range(args.start, args.end + 1):
        print(f"  scanning id {motor_id}...", end="\r", flush=True)
        try:
            pos = ctrl.sync_read_present_position([motor_id])
            angle_rad = pos[0]
            found.append((motor_id, angle_rad))
            print(f"  [OK]   id {motor_id:3d}: {math.degrees(angle_rad):7.2f}°  "
                  f"({angle_rad:+.3f} rad)")
        except RuntimeError:
            # No reply — ID not present. Skip silently.
            pass

    elapsed = time.monotonic() - t0
    # Clear progress line.
    print(" " * 40, end="\r")

    print(f"\nDone in {elapsed:.1f}s — {len(found)} motor(s) found: "
          f"{[m for m, _ in found]}")

    if not found:
        sys.exit(1)


if __name__ == "__main__":
    main()
