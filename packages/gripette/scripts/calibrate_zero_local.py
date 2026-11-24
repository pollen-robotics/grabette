"""Local zero-calibration: record encoder offsets at the user-defined zero.

Workflow:
    1. Stop the gripette service so the serial bus is free:
           sudo systemctl stop gripette
    2. Run this script.
    3. With motors torque-off, manually move the gripper to its TRUE zero
       (fully open, robot frame = 0).
    4. Confirm. The script first re-centres each servo's own zero on this
       pose (EEPROM offset register), so the closing travel never runs into
       the encoder's 0/4095 seam, then reads the raw encoder positions and
       writes them as GRIPPER_MOTOR1_OFFSET / GRIPPER_MOTOR2_OFFSET into
       /etc/gripette/env, preserving any other lines in the file (notably
       GRIPPER_HAND). Offsets that would still cap the closing travel below
       the joint limits are refused.
    5. Restart the service:
           sudo systemctl start gripette

The MotorController is instantiated WITHOUT sign/offset corrections so that
read_positions() returns raw encoder values — that's what we want to capture
as the offset.

Usage:
    uv run python scripts/calibrate_zero_local.py
    uv run python scripts/calibrate_zero_local.py --yes        # skip prompts
    uv run python scripts/calibrate_zero_local.py --dry-run    # print, don't write
    uv run python scripts/calibrate_zero_local.py --no-center  # keep the servo zero as is
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from gripette.config import settings
from gripette.hardware.motors import MotorController, travel_problem

ENV_FILE = Path("/etc/gripette/env")
SAMPLES = 10  # average a few readings to suppress encoder jitter
STALE_WARNING = (
    "WARNING: the servo zero WAS re-centred, so the offsets currently in "
    f"{ENV_FILE} no longer match the motors. Do not start the gripette "
    "service before running this calibration to the end."
)


def read_raw_encoder(motors: MotorController) -> tuple[float, float]:
    """Average a few cached reads to suppress jitter."""
    # Give the bus loop time to populate the cache.
    time.sleep(0.2)
    sum1 = sum2 = 0.0
    for _ in range(SAMPLES):
        m1, m2 = motors.read_positions()
        sum1 += m1
        sum2 += m2
        time.sleep(0.05)
    return (sum1 / SAMPLES, sum2 / SAMPLES)


def device_signs(path: Path) -> tuple[int, int]:
    """Robot <-> encoder signs the service will run with: the hand (and any
    sign override) from the env file, falling back to this shell's settings.
    The script itself runs with an identity transform, so it can't use those."""
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)=(.*)$", line)
            if m:
                env[m.group(1)] = m.group(2).strip()
    hand = env.get("GRIPPER_HAND", settings.hand)
    default = (+1, +1) if hand == "right" else (-1, -1)
    return (int(env.get("GRIPPER_MOTOR1_SIGN", default[0])),
            int(env.get("GRIPPER_MOTOR2_SIGN", default[1])))


def merge_env_file(path: Path, new_offsets: dict[str, str]) -> str:
    """Return updated env-file content with new_offsets merged in.

    Any existing GRIPPER_MOTOR{1,2}_OFFSET lines are replaced; everything
    else (HAND, custom signs, comments, blank lines) is preserved verbatim.
    """
    keys = set(new_offsets.keys())
    lines_out = []
    seen_keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"^\s*([A-Z_][A-Z0-9_]*)=", line)
            if m and m.group(1) in keys:
                # Replace this line with the new value.
                lines_out.append(f"{m.group(1)}={new_offsets[m.group(1)]}")
                seen_keys.add(m.group(1))
            else:
                lines_out.append(line)
    # Append any keys we didn't see (first-time calibration).
    for k, v in new_offsets.items():
        if k not in seen_keys:
            lines_out.append(f"{k}={v}")
    if lines_out and lines_out[-1] != "":
        lines_out.append("")
    return "\n".join(lines_out)


def write_env_with_sudo(path: Path, content: str) -> None:
    """Write content to a root-owned path via 'sudo tee'."""
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent.exists() else None
    # Ensure /etc/gripette exists; mkdir needs sudo too.
    subprocess.run(["sudo", "mkdir", "-p", str(path.parent)], check=True)
    proc = subprocess.run(
        ["sudo", "tee", str(path)],
        input=content.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sudo tee {path} failed: {proc.stderr.decode('utf-8', errors='replace')}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the 'apply?' confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written, but don't touch the env file "
                             "or the servos (implies --no-center).")
    parser.add_argument("--no-center", action="store_true",
                        help="Don't re-centre the servos' own zero (EEPROM) on the open "
                             "pose; only record the encoder readings.")
    args = parser.parse_args()

    # Identity controller (no sign/offset) so read_positions = raw encoder.
    motors = MotorController(
        port=settings.motor_port,
        baudrate=settings.motor_baudrate,
        id_1=settings.motor_id_1,
        id_2=settings.motor_id_2,
        signs=(1, 1),
        offsets=(0.0, 0.0),
    )

    print(f"Starting motors on {settings.motor_port} (identity transform)...")
    try:
        motors.start()
    except Exception as e:
        print(f"ERROR: could not open the motor bus: {e}", file=sys.stderr)
        print("Is gripette.service running? Stop it first:", file=sys.stderr)
        print("  sudo systemctl stop gripette", file=sys.stderr)
        sys.exit(1)

    motors.set_torque(False)
    print("Torque OFF — gripper is back-drivable.\n")
    print("Move the gripper to the TRUE zero position (fully open).")
    if not args.yes:
        try:
            input("Press ENTER when the gripper is at zero (Ctrl-C to cancel): ")
        except (KeyboardInterrupt, EOFError):
            print("\nCanceled.")
            motors.stop()
            sys.exit(1)

    centred = False
    if not (args.dry_run or args.no_center):
        print("\nRe-centring the servo zero on this pose (keep the gripper still)...")
        try:
            r = motors.center_servo_offsets()
        except RuntimeError as e:
            motors.stop()
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        centred = r["changed"]
        if centred:
            print(f"  servo offsets {r['old_offsets']} -> {r['new_offsets']} steps")
        else:
            print("  already centred, nothing written")

    print(f"\nReading encoder positions (averaging {SAMPLES} samples)...")
    m1, m2 = read_raw_encoder(motors)
    motors.stop()

    print(f"  m1 encoder = {m1:+.4f} rad")
    print(f"  m2 encoder = {m2:+.4f} rad")
    print()

    problem = travel_problem((m1, m2), device_signs(ENV_FILE),
                             (settings.motor1_max, settings.motor2_max))
    if problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        if centred:
            print(STALE_WARNING, file=sys.stderr)
        if args.dry_run or args.no_center:
            print("Run again without --dry-run / --no-center to re-centre.",
                  file=sys.stderr)
        sys.exit(1)

    new_offsets = {
        "GRIPPER_MOTOR1_OFFSET": f"{m1:.6f}",
        "GRIPPER_MOTOR2_OFFSET": f"{m2:.6f}",
    }

    print(f"Will write to {ENV_FILE} (preserving any other lines):")
    new_content = merge_env_file(ENV_FILE, new_offsets)
    for line in new_content.splitlines():
        marker = "  >> " if line.startswith("GRIPPER_MOTOR") else "     "
        print(f"{marker}{line}")
    print()

    if args.dry_run:
        print("--dry-run: not writing.")
        return

    if not args.yes:
        try:
            answer = input("Apply? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = ""
        if answer not in ("y", "yes"):
            print("\nAborted.")
            if centred:
                print(STALE_WARNING, file=sys.stderr)
            return

    write_env_with_sudo(ENV_FILE, new_content)
    print(f"\nWrote {ENV_FILE}.")
    print("Restart the service for changes to take effect:")
    print("  sudo systemctl restart gripette")


if __name__ == "__main__":
    main()
