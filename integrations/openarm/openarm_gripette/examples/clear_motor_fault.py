"""Clear a Damiao motor's LATCHED FAULT (red LED) directly over CAN.

After a hard impact a motor can latch a protection fault (over-current /
stall) that survives detection scans — the motor shows a RED LED, ignores
enable/position commands, and may not answer some queries. The DM protocol's
clear-error command ([0xFF]*7 + 0xFB — same family as enable 0xFC / disable
0xFD) resets the latch without a power cycle.

This sends clear-error to the chosen motor(s), then health-checks with the
REFRESH command (answered even by disabled motors).

Run ON the CAN-connected machine with the arm server STOPPED:

  uv run python examples/clear_motor_fault.py --can_port can0 --motor 7
  uv run python examples/clear_motor_fault.py --can_port can0            # all 7

If the LED stays red after this + a full power cycle, the fault is
re-triggering on boot (mechanical jam — rotate the joint by hand, power off,
and feel for grinding) or the motor is damaged (encoder/driver).
"""

import argparse
import time

import can

MOTOR_IDS = list(range(0x01, 0x08))
MASTER_TO_JOINT = {0x10 + i: f"joint_{i}" for i in range(1, 8)}
CLEAR_FRAME = [0xFF] * 7 + [0xFB]
CAN_PARAM_ID = 0x7FF
CAN_CMD_REFRESH = 0xCC


def parse_args():
    p = argparse.ArgumentParser(description="Clear latched Damiao motor fault(s) over CAN")
    p.add_argument("--can_port", type=str, default="can0")
    p.add_argument("--motor", type=int, default=None,
                   help="Joint number 1-7 (default: send clear to all)")
    p.add_argument("--no_fd", action="store_true",
                   help="Classic CAN frames (OpenArm default is CAN FD)")
    return p.parse_args()


def drain(bus):
    while bus.recv(timeout=0.0) is not None:
        pass


def main():
    args = parse_args()
    use_fd = not args.no_fd
    targets = [args.motor] if args.motor else MOTOR_IDS
    bus = can.interface.Bus(channel=args.can_port, interface="socketcan", fd=use_fd)
    try:
        for pass_idx in range(3):
            for motor_id in targets:
                bus.send(can.Message(arbitration_id=motor_id, data=CLEAR_FRAME,
                                     is_extended_id=False, is_fd=use_fd))
                time.sleep(0.005)
            time.sleep(0.1)
        print(f"clear-error sent x3 to joint(s) {targets}")

        drain(bus)
        responding = set()
        for motor_id in MOTOR_IDS:
            data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
            bus.send(can.Message(arbitration_id=CAN_PARAM_ID, data=data,
                                 is_extended_id=False, is_fd=use_fd))
            deadline = time.monotonic() + 0.05
            while time.monotonic() < deadline:
                msg = bus.recv(timeout=0.01)
                if msg is not None and msg.arbitration_id in MASTER_TO_JOINT:
                    responding.add(MASTER_TO_JOINT[msg.arbitration_id])
        missing = [j for j in MASTER_TO_JOINT.values() if j not in responding]
        print(f"responding: {sorted(responding)}")
        if missing:
            print(f"STILL SILENT: {missing} — if its LED is red after this, power-cycle; "
                  f"if red persists, suspect mechanical jam or damaged encoder/driver.")
    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()
