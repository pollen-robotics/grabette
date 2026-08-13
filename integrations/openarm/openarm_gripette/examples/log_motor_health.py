"""Passively log DM motor health (fault status + temperatures) to a CSV.

The motors keep NO fault history: the status nibble in each feedback frame
(byte 0 high nibble, see read_motor_fault.py) is the only observability and
it is volatile. This logger builds the missing history on the host side.

RX-only by construction: it opens its own socketcan socket (the kernel gives
every socket a copy of all frames, nothing is stolen from the arm server) and
never sends a frame. It decodes the feedback frames the server's control loop
already elicits (master IDs 0x11-0x17) and records:

  - every status-nibble TRANSITION per joint (enabled/disabled/fault codes)
  - temperatures at 1 Hz per joint
  - a joint going SILENT while the rest of the bus is still streaming
    (the observed joint-7 failure mode), and its reappearance

Start it in a separate terminal before an eval/teleop session:

  uv run python examples/log_motor_health.py --can_port can0
  uv run python examples/log_motor_health.py --out /tmp/health.csv

Stop with Ctrl-C. Idle bus (server stopped) logs nothing — motors only emit
feedback when commanded.
"""

import argparse
import csv
import datetime
import time

import can

MASTER_TO_JOINT = {0x10 + i: i for i in range(1, 8)}
TEMP_PERIOD_S = 1.0
SILENCE_S = 2.0

STATUS = {
    0x0: "disabled (no fault)",
    0x1: "enabled (no fault)",
    0x8: "FAULT: overvoltage",
    0x9: "FAULT: undervoltage",
    0xA: "FAULT: overcurrent",
    0xB: "FAULT: MOS overtemperature",
    0xC: "FAULT: motor coil overtemperature",
    0xD: "FAULT: communication loss (CAN timeout)",
    0xE: "FAULT: overload",
}


def parse_args():
    p = argparse.ArgumentParser(description="Passively log DM motor fault status + temps to CSV")
    p.add_argument("--can_port", type=str, default="can0")
    p.add_argument("--out", type=str, default=None,
                   help="CSV path (default: motor_health_<timestamp>.csv in cwd)")
    p.add_argument("--no_fd", action="store_true",
                   help="Classic CAN frames (OpenArm default is CAN FD)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or datetime.datetime.now().strftime("motor_health_%Y%m%d_%H%M%S.csv")
    filters = [{"can_id": mid, "can_mask": 0x7FF} for mid in MASTER_TO_JOINT]
    bus = can.interface.Bus(channel=args.can_port, interface="socketcan",
                            fd=not args.no_fd, can_filters=filters)

    f = open(out_path, "w", newline="", buffering=1)
    writer = csv.writer(f)
    writer.writerow(["unix_time", "iso_time", "joint", "event",
                     "status", "status_label", "t_mos_c", "t_rotor_c"])

    def emit(joint, event, status=None, t_mos=None, t_rotor=None, echo=False):
        now = time.time()
        label = STATUS.get(status, f"unknown 0x{status:X}") if status is not None else ""
        writer.writerow([f"{now:.3f}", datetime.datetime.fromtimestamp(now).isoformat(),
                         joint, event,
                         f"0x{status:X}" if status is not None else "", label,
                         t_mos if t_mos is not None else "",
                         t_rotor if t_rotor is not None else ""])
        if echo:
            print(f"[{datetime.datetime.fromtimestamp(now).strftime('%H:%M:%S')}] "
                  f"joint_{joint} {event}"
                  + (f": {label} (T_mos={t_mos}C T_rotor={t_rotor}C)" if status is not None else ""))

    last_status = {}     # joint -> last seen status nibble
    last_temp = {}       # joint -> monotonic time of last periodic temp row
    last_seen = {}       # joint -> monotonic time of last frame
    silent = set()

    print(f"logging to {out_path} (RX-only on {args.can_port}, Ctrl-C to stop)")
    try:
        while True:
            msg = bus.recv(timeout=0.5)
            mono = time.monotonic()
            if msg is not None and len(msg.data) >= 8:
                joint = MASTER_TO_JOINT[msg.arbitration_id]
                status = msg.data[0] >> 4
                t_mos, t_rotor = msg.data[6], msg.data[7]
                if joint in silent:
                    silent.discard(joint)
                    emit(joint, "reappeared", status, t_mos, t_rotor, echo=True)
                last_seen[joint] = mono
                if status != last_status.get(joint):
                    last_status[joint] = status
                    emit(joint, "status_change", status, t_mos, t_rotor, echo=True)
                elif mono - last_temp.get(joint, 0.0) >= TEMP_PERIOD_S:
                    last_temp[joint] = mono
                    emit(joint, "temp", status, t_mos, t_rotor)
            # a joint is "silent" only if OTHERS are still streaming (bus active);
            # an all-quiet bus just means the server is stopped
            bus_active = any(mono - t < SILENCE_S for t in last_seen.values())
            if bus_active:
                for joint, t in last_seen.items():
                    if mono - t >= SILENCE_S and joint not in silent:
                        silent.add(joint)
                        emit(joint, "SILENT (no feedback while bus active)",
                             last_status.get(joint), echo=True)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        f.close()
        bus.shutdown()


if __name__ == "__main__":
    main()
