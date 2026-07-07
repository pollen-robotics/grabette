"""Configure a single Feetech STS3215 servo for assembly into a gripette.

Brand-new motors ship from the factory as ID=1, baudrate=1 Mbps (index 0),
position-control mode. A gripette needs one motor at ID=1 (proximal) and
one at ID=2 (distal), so for each new gripper one of the two motors must
have its ID changed from 1 → 2.

  +-------------+----------+-------+
  | role        | motor_id | side  |
  +-------------+----------+-------+
  | proximal    |     1    | base  |
  | distal      |     2    | tip   |
  +-------------+----------+-------+

Workflow (per motor):
  1. Plug ONE motor at a time onto the bus — both motors at ID=1 will
     collide and the bus will return nothing usable.
  2. Run this script. It scans IDs 1..253 to find the motor and prints
     its current config.
  3. Pick the target role; the script writes the new ID (with the
     EEPROM unlock/lock dance) and verifies by re-reading.
  4. PHYSICALLY LABEL the motor before unplugging it ("P" or "D"), then
     repeat with the other motor.

Usage:
    python scripts/configure_motor.py                    # interactive
    python scripts/configure_motor.py --role proximal    # non-interactive
    python scripts/configure_motor.py --info             # read-only, no writes
    python scripts/configure_motor.py --port /dev/ttyUSB0 --baudrate 1000000
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

# Role → motor ID. The gripette runtime hard-codes this mapping
# (see teleop_bridge.py: m1=proximal, m2=distal; config.py: motor_id_{1,2}={1,2}).
ROLE_TO_ID = {"proximal": 1, "distal": 2}
ID_TO_ROLE = {v: k for k, v in ROLE_TO_ID.items()}

# Factory defaults we expect to see on a brand-new STS3215. Mismatches don't
# block configuration but the user is warned.
EXPECTED_BAUDRATE_IDX = 0  # 0 = 1 Mbps (the STS3215 default and the one gripette uses)
EXPECTED_MODE = 0          # 0 = position, 1 = wheel, 2 = PWM, 3 = step

MODE_NAMES = {0: "position", 1: "wheel", 2: "PWM", 3: "step"}
# Feetech STS3215 baudrate index → bps. We only verify index 0 in practice.
BAUDRATE_TABLE = {0: 1_000_000, 1: 500_000, 2: 250_000, 3: 128_000,
                  4: 115_200, 5: 76_800, 6: 57_600, 7: 38_400}

# Short per-ID timeout for the scan: STS3215 replies in <2 ms at 1 Mbps,
# so 30 ms is generous and keeps a full sweep under ~8 s of dead air.
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


def scan_bus(ctrl: Sts3215PyController, start: int = 1, end: int = 253) -> list[int]:
    """Return IDs that reply on the bus. Uses ping (no register read) for speed."""
    found = []
    for motor_id in range(start, end + 1):
        print(f"  scanning id {motor_id}...", end="\r", flush=True)
        try:
            if ctrl.ping(motor_id):
                found.append(motor_id)
        except RuntimeError:
            pass
    print(" " * 40, end="\r")
    return found


def read_config(ctrl: Sts3215PyController, motor_id: int) -> dict:
    """Read the registers we care about. All read_* methods return a 1-element list."""
    return {
        "id":          ctrl.read_id(motor_id)[0],
        "baudrate":    ctrl.read_baudrate(motor_id)[0],
        "mode":        ctrl.read_mode(motor_id)[0],
        "lock":        ctrl.read_lock(motor_id)[0],
        "position":    ctrl.read_present_position(motor_id)[0],   # radians
        "voltage_dV":  ctrl.read_present_voltage(motor_id)[0],    # tenths of a volt
        "temp_C":      ctrl.read_present_temperature(motor_id)[0],
    }


def print_config(cfg: dict) -> None:
    bps = BAUDRATE_TABLE.get(cfg["baudrate"], f"unknown idx {cfg['baudrate']}")
    mode_name = MODE_NAMES.get(cfg["mode"], f"unknown ({cfg['mode']})")
    role = ID_TO_ROLE.get(cfg["id"], "—")
    print(f"  ID            : {cfg['id']}  (role: {role})")
    print(f"  baudrate idx  : {cfg['baudrate']}  ({bps} bps)")
    print(f"  mode          : {cfg['mode']}  ({mode_name})")
    print(f"  EEPROM lock   : {'locked' if cfg['lock'] else 'unlocked'}")
    print(f"  position      : {math.degrees(cfg['position']):+7.2f}°  "
          f"({cfg['position']:+.3f} rad)")
    print(f"  voltage       : {cfg['voltage_dV'] / 10:.1f} V")
    print(f"  temperature   : {cfg['temp_C']} °C")


def warn_if_unexpected(cfg: dict) -> None:
    """Surface anything that differs from a stock STS3215 — not a hard failure."""
    if cfg["baudrate"] != EXPECTED_BAUDRATE_IDX:
        print(f"  WARNING: baudrate idx {cfg['baudrate']} ≠ expected "
              f"{EXPECTED_BAUDRATE_IDX} (1 Mbps). gripette runs at 1 Mbps; "
              f"this motor will be unreachable in the runtime.")
    if cfg["mode"] != EXPECTED_MODE:
        print(f"  WARNING: mode {cfg['mode']} ≠ expected {EXPECTED_MODE} (position). "
              f"gripette uses position control.")
    if not (60 <= cfg["voltage_dV"] <= 90):
        print(f"  WARNING: voltage {cfg['voltage_dV'] / 10:.1f} V outside the "
              f"nominal 6.0–9.0 V range — check the power supply.")
    if cfg["temp_C"] > 50:
        print(f"  WARNING: temperature {cfg['temp_C']} °C is high — let it cool.")


def change_id(ctrl: Sts3215PyController, current_id: int, new_id: int) -> bool:
    """Unlock EEPROM -> write ID -> verify at new ID -> lock EEPROM.

    Returns True on success.

    Note on the write_id timeout: Feetech firmwares commonly send the status
    reply for an ID-write *from the new ID*, which the host times out waiting
    on the old ID. The write itself succeeds — we treat the timeout as
    expected and verify by reading at the new ID immediately after.
    """
    print(f"  unlocking EEPROM on id {current_id}...")
    ctrl.write_lock(current_id, False)

    print(f"  writing new id: {current_id} -> {new_id}...")
    try:
        ctrl.write_id(current_id, new_id)
    except RuntimeError:
        # Expected — status ACK comes from the new ID. Verify by read below.
        print("  (status ACK missed — expected for ID writes; verifying...)")

    # Settle: ~100 ms for the motor to commit and start replying at the new ID.
    time.sleep(0.1)

    try:
        readback = ctrl.read_id(new_id)[0]
    except RuntimeError as e:
        print(f"  ERROR: motor does not respond at new id {new_id}: {e}")
        print(f"         The ID write did not take effect. Power-cycle the motor")
        print(f"         and re-run; if it persists at id {current_id}, EEPROM may still be locked.")
        return False
    if readback != new_id:
        print(f"  ERROR: read-back returned {readback}, expected {new_id}")
        return False
    print(f"  verified: motor now responds at id {new_id}")

    print(f"  locking EEPROM at new id {new_id}...")
    try:
        ctrl.write_lock(new_id, True)
    except RuntimeError as e:
        print(f"  WARNING: could not lock EEPROM at id {new_id}: {e}")
        print(f"           ID change succeeded; EEPROM left unlocked (harmless,")
        print(f"           but other EEPROM regs will be writable until next power cycle).")

    return True


def prompt_role() -> str:
    if not sys.stdin.isatty():
        print("ERROR: not a TTY — pass --role proximal|distal for non-interactive use.",
              file=sys.stderr)
        sys.exit(1)
    while True:
        choice = input("Target role? [p]roximal (id=1) / [d]istal (id=2) / [q]uit: ").strip().lower()
        if choice in ("p", "proximal"):
            return "proximal"
        if choice in ("d", "distal"):
            return "distal"
        if choice in ("q", "quit", ""):
            print("Aborted.")
            sys.exit(0)


def confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        return False
    return input(f"{prompt} [y/N]: ").strip().lower() in ("y", "yes")


def main():
    parser = argparse.ArgumentParser(
        description="Configure a single STS3215 servo: scan, inspect, and set its ID.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Roles: proximal -> id=1, distal -> id=2.",
    )
    parser.add_argument("--port", default=settings.motor_port,
                        help=f"Serial port (default: {settings.motor_port})")
    parser.add_argument("--baudrate", type=int, default=settings.motor_baudrate,
                        help=f"Baudrate (default: {settings.motor_baudrate})")
    parser.add_argument("--role", choices=["proximal", "distal"],
                        help="Target role. Skips the interactive prompt.")
    parser.add_argument("--info", action="store_true",
                        help="Read-only: print current config and exit, no writes.")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the 'apply changes?' confirmation.")
    args = parser.parse_args()

    print(f"Opening {args.port} @ {args.baudrate} baud...")
    flush_serial(args.port, args.baudrate)
    ctrl = Sts3215PyController(args.port, args.baudrate, SCAN_TIMEOUT)

    print("Scanning bus for connected motor (1..253)...")
    found = scan_bus(ctrl)

    if not found:
        print("ERROR: no motors responded.", file=sys.stderr)
        print("  - check the power supply (motor LED should blink on startup)", file=sys.stderr)
        print("  - check the serial wiring (TX/RX swap, GND continuity)", file=sys.stderr)
        print(f"  - check the baudrate (default: {settings.motor_baudrate})", file=sys.stderr)
        print("  - on a Pi, verify the serial console is disabled (make enable-uart)", file=sys.stderr)
        sys.exit(1)

    if len(found) > 1:
        print(f"ERROR: {len(found)} motors found on the bus: {found}", file=sys.stderr)
        print("       Disconnect all but ONE motor and re-run.", file=sys.stderr)
        print("       (Brand-new motors all share ID=1, so their replies collide.)",
              file=sys.stderr)
        sys.exit(1)

    current_id = found[0]
    print(f"Found 1 motor at id {current_id}. Reading config...\n")
    cfg = read_config(ctrl, current_id)
    print_config(cfg)
    warn_if_unexpected(cfg)
    print()

    if args.info:
        return

    target_role = args.role or prompt_role()
    target_id = ROLE_TO_ID[target_role]

    if current_id == target_id:
        print(f"Motor is already at id {target_id} ({target_role}). Nothing to do.")
        print(f"Label this motor as '{target_role[0].upper()}' before unplugging.")
        return

    print(f"About to change id {current_id} -> {target_id} ({target_role}).")
    if not args.yes and not confirm("Apply?"):
        print("Aborted.")
        return

    if not change_id(ctrl, current_id, target_id):
        sys.exit(1)

    print()
    print(f"Done. This motor is now {target_role} (id={target_id}).")
    print(f"*** LABEL IT '{target_role[0].upper()}' ON THE BODY before unplugging. ***")


if __name__ == "__main__":
    main()
