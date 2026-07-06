"""Software e-stop + CAN health check: disable all arm motors DIRECTLY on the
CAN bus, no gRPC server needed.

DANGER: any motor still powered freewheels when disabled — the arm FALLS.

Use when the server died (or was killed) and a motor kept holding torque:
the LeRobot/Damiao disable is a single fire-and-forget frame per motor, so a
lost frame silently leaves that motor powered. This script sends several full
disable passes, then health-checks each motor with the REFRESH command.

Note: motors do NOT reliably ACK a disable (an already-disabled motor stays
silent), so the disable passes are fire-and-forget. The refresh command is
answered even by disabled motors — that's the health check: a motor that
doesn't answer refresh has a communication problem (wiring, connector, ID
config) or is unpowered.

The OpenArm bus runs CAN FD (LeRobot `use_can_fd=True`, data bitrate 5 Mbps).
Frames sent as classic CAN are IGNORED by the motors, and a non-FD socket
can't receive their FD replies — this script must mirror the driver's FD
settings exactly. Pass --no_fd only if your bus is genuinely classic CAN.

Run ON the CAN-connected machine, with the server STOPPED (both talking on
the same bus confuses response matching):

  uv run python examples/disable_all_motors_can.py --can_port can0
"""

import argparse
import time

import can

# Arm motor send IDs and their feedback (master) IDs — same as the calibration
# script and the LeRobot config: joint_1..joint_7 = 0x01..0x07 / 0x11..0x17.
MOTOR_IDS = list(range(0x01, 0x08))
MASTER_TO_JOINT = {0x10 + i: f"joint_{i}" for i in range(1, 8)}
DISABLE_FRAME = [0xFF] * 7 + [0xFD]
# Refresh: sent on the param ID, addressed to the motor in the payload
# (LeRobot damiao driver convention: CAN_PARAM_ID=0x7FF, CAN_CMD_REFRESH=0xCC).
CAN_PARAM_ID = 0x7FF
CAN_CMD_REFRESH = 0xCC


def parse_args():
    p = argparse.ArgumentParser(description="Disable all arm motors directly over CAN (arm falls!)")
    p.add_argument("--can_port", type=str, default="can0", help="CAN interface name")
    p.add_argument("--passes", type=int, default=3, help="Number of full disable passes")
    p.add_argument("--no_fd", action="store_true",
                   help="Use classic CAN frames (OpenArm default is CAN FD — only for non-FD buses)")
    return p.parse_args()


def drain(bus):
    while bus.recv(timeout=0.0) is not None:
        pass


def main():
    args = parse_args()
    use_fd = not args.no_fd
    print("Disabling ALL arm motors — any joint still powered will FALL.")
    print(f"CAN mode: {'FD (matches LeRobot use_can_fd=True)' if use_fd else 'classic'}")

    # Same socket settings as the LeRobot damiao driver (bitrate is set at the
    # `ip link` level for socketcan; fd=True enables sending/receiving FD frames).
    bus = can.interface.Bus(channel=args.can_port, interface="socketcan", fd=use_fd)
    try:
        # Fire-and-forget disable passes (no ACK expected).
        for pass_idx in range(1, args.passes + 1):
            for motor_id in MOTOR_IDS:
                bus.send(can.Message(arbitration_id=motor_id, data=DISABLE_FRAME,
                                     is_extended_id=False, is_fd=use_fd))
                time.sleep(0.005)
            print(f"disable pass {pass_idx}/{args.passes} sent")
            time.sleep(0.1)

        # Health check: refresh each motor; disabled motors DO answer this.
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
        if missing:
            print(f"\nWARNING: no refresh response from {missing} — those motors have a "
                  "communication problem (check wiring/connectors) or are unpowered. "
                  "They may NOT have received the disable either!")
        else:
            print("\nAll 7 motors answered the refresh — bus healthy, disable frames "
                  "were delivered; the whole arm should be torque-free.")
    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()
