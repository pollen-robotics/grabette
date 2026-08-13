"""Read a Damiao motor's latched fault code (why the LED is red) over CAN.

Every DM feedback frame carries a status nibble in the high 4 bits of byte 0
(the low 4 bits are the controller ID). The REFRESH command (0x7FF / 0xCC)
makes a motor emit one feedback frame even when disabled or faulted, so we
can decode WHICH protection tripped instead of blind-clearing:

  0x0 disabled (no fault)      0xB MOS overtemperature
  0x1 enabled  (no fault)      0xC motor coil overtemperature
  0x8 overvoltage              0xD communication loss (CAN timeout)
  0x9 undervoltage             0xE overload
  0xA overcurrent

Bytes 6/7 of the same frame are the MOS and rotor temperatures (degC).

Read-only: sends REFRESH queries only — no enable, no clear, no motion.
Run ON the CAN-connected machine with the arm server STOPPED:

  uv run python examples/read_motor_fault.py --can_port can0
"""

import argparse
import time

import can

MOTOR_IDS = list(range(0x01, 0x08))
MASTER_TO_JOINT = {0x10 + i: i for i in range(1, 8)}
CAN_PARAM_ID = 0x7FF
CAN_CMD_REFRESH = 0xCC

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
    p = argparse.ArgumentParser(description="Read latched Damiao motor fault code(s) over CAN")
    p.add_argument("--can_port", type=str, default="can0")
    p.add_argument("--motor", type=int, default=None,
                   help="Joint number 1-7 (default: query all)")
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
        drain(bus)
        for motor_id in targets:
            data = [motor_id & 0xFF, (motor_id >> 8) & 0xFF, CAN_CMD_REFRESH, 0, 0, 0, 0, 0]
            bus.send(can.Message(arbitration_id=CAN_PARAM_ID, data=data,
                                 is_extended_id=False, is_fd=use_fd))
            frame = None
            deadline = time.monotonic() + 0.1
            while time.monotonic() < deadline:
                msg = bus.recv(timeout=0.02)
                if msg is not None and MASTER_TO_JOINT.get(msg.arbitration_id) == motor_id:
                    frame = msg
                    break
            if frame is None:
                print(f"joint_{motor_id}: NO RESPONSE (not powered / CAN wiring / wrong ID)")
                continue
            status = frame.data[0] >> 4
            t_mos, t_rotor = frame.data[6], frame.data[7]
            label = STATUS.get(status, f"unknown status 0x{status:X}")
            print(f"joint_{motor_id}: {label}  [T_mos={t_mos}C T_rotor={t_rotor}C "
                  f"raw={frame.data.hex(' ')}]")
    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()
