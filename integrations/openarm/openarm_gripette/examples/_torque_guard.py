"""Shared abort handler for the motion example scripts.

Every arm-moving example calls `abort_torque_off(stub, args.keep_torque)` from
its Ctrl+C / crash handlers: torque is disabled so the arm never keeps pushing
after an aborted run. The motors then freewheel and the arm FALLS under
gravity — that is the intended fail-safe. Normal completion leaves torque on
(arm keeps holding). Pass --keep_torque to opt out.

In simulation SetTorque is a no-op, so the same scripts work unchanged.
"""

import logging

from openarm_gripette_simu.proto import arm_pb2

logger = logging.getLogger(__name__)


def abort_torque_off(stub, keep_torque: bool = False):
    """Best-effort arm torque disable after an aborted run (Ctrl+C or crash)."""
    if keep_torque:
        logger.warning("Aborted — torque left ON (--keep_torque).")
        return
    try:
        stub.SetTorque(arm_pb2.SetTorqueRequest(enable=False), timeout=2.0)
        logger.warning("Aborted — arm torque DISABLED (arm falls under gravity).")
    except Exception as e:
        logger.error(f"Aborted, and disabling torque FAILED: {e} — arm may still be powered!")


def add_keep_torque_arg(parser):
    parser.add_argument(
        "--keep_torque",
        action="store_true",
        help="On Ctrl+C / crash, leave torque ON (default: disable torque, arm falls)",
    )
