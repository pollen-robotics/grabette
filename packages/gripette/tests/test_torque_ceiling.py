"""The torque ceiling is what makes the raised position limits safe.

motor1_max is now the measured collision angle, so a full close deliberately
drives into the mechanical stop and relies on torque to halt it. At 100% effort
that same command grinds into the collision. Before this, an unset per-command
limit meant "leave the current limit alone" — i.e. full torque on a fresh boot —
so the protection depended on every client remembering to pass a limit.

These tests pin the guarantee: no path through the servicer can exceed the
ceiling.
"""

import sys
import types

import pytest

# gripette.service imports hardware.motors, which imports `serial` (pyserial) —
# a Pi-only dependency absent on a workstation and in CI. Stub it so the REAL
# servicer code is exercised rather than a reimplementation of its logic. Nothing
# in these tests touches a serial port; the motors object is a fake.
if "serial" not in sys.modules:
    import importlib.machinery

    _stub = types.ModuleType("serial")
    # A real __spec__ is required: without it, anything that later inspects the
    # module through importlib raises "serial.__spec__ is None" — which broke 17
    # unrelated tests once this stub leaked across a shared pytest session.
    _stub.__spec__ = importlib.machinery.ModuleSpec("serial", None)
    _stub.Serial = object          # referenced by motors.py for typing/construction
    _stub.SerialException = Exception
    sys.modules["serial"] = _stub

from gripette.config import settings  # noqa: E402
from gripette.proto import gripper_pb2  # noqa: E402
from gripette.service import GripperServicer  # noqa: E402

CEILING = settings.torque_ceiling


class FakeMotors:
    def __init__(self):
        self.limit_writes = []
        self.goals = []

    def set_torque_limit(self, values):
        self.limit_writes.append(tuple(values))

    def write_goal_positions(self, p, d):
        self.goals.append((p, d))

    def read_positions(self):
        return 0.0, 0.0

    def get_present_load(self):
        return 0.0, 0.0


def make():
    motors = FakeMotors()
    servicer = GripperServicer(camera=object(), motors=motors, sync=object())
    return servicer, motors


def send(servicer, m1=0.0, m2=0.0, t1=0.0, t2=0.0):
    return servicer.SendMotorCommand(
        gripper_pb2.MotorCommand(motor1_goal=m1, motor2_goal=m2,
                                 motor1_torque_limit=t1, motor2_torque_limit=t2),
        None,
    )


def test_ceiling_is_applied_at_startup():
    """Between SetTorque(enable) and the first goal, the servo must already be
    capped — otherwise it holds whatever limit it had stored."""
    _servicer, motors = make()
    assert motors.limit_writes == [(CEILING, CEILING)]


def test_unset_limit_resolves_to_the_ceiling_not_full_torque():
    """The regression this exists for: 0 means "unset" in the proto, and used to
    mean "leave alone" = full torque."""
    servicer, motors = make()
    motors.limit_writes.clear()
    servicer._last_torque_limit = None
    r = send(servicer, m1=1.0, t1=0.0, t2=0.0)
    assert r.success
    assert motors.limit_writes == [(CEILING, CEILING)]


def test_a_request_above_the_ceiling_is_clamped_not_rejected():
    """A caller asking for more effort than allowed should still get a working
    grasp, just a capped one.

    Asserts the EFFECTIVE limit rather than the write list: startup already put
    the ceiling in place, so clamping 1.0 down to it is correctly deduped into no
    write at all. What matters is the value in force, not the traffic.
    """
    servicer, _motors = make()
    r = send(servicer, m1=1.0, t1=1.0, t2=1.0)
    assert r.success
    assert servicer._last_torque_limit == pytest.approx((CEILING, CEILING))


def test_a_request_below_the_ceiling_passes_through():
    """0.25 is the value the field grasps used; it must survive untouched.
    approx because the proto field is float32, so 0.25-class values round-trip."""
    servicer, motors = make()
    motors.limit_writes.clear()
    r = send(servicer, m1=1.0, t1=0.25, t2=0.25)
    assert r.success
    assert len(motors.limit_writes) == 1
    assert motors.limit_writes[0] == pytest.approx((0.25, 0.25), abs=1e-6)


def test_mixed_request_is_capped_per_motor():
    servicer, motors = make()
    motors.limit_writes.clear()
    send(servicer, m1=1.0, t1=0.2, t2=0.99)
    assert len(motors.limit_writes) == 1
    assert motors.limit_writes[0] == pytest.approx((0.2, CEILING), abs=1e-6)


def test_repeated_identical_limits_are_written_once():
    """Deduped so a constant per-command value isn't re-deposited at stream rate."""
    servicer, motors = make()
    motors.limit_writes.clear()
    for _ in range(5):
        send(servicer, m1=1.0, t1=0.25, t2=0.25)
    assert len(motors.limit_writes) == 1


def test_goals_are_still_forwarded():
    """The ceiling must not interfere with the actual command path."""
    servicer, motors = make()
    send(servicer, m1=1.2, m2=0.3, t1=0.25, t2=0.25)
    assert motors.goals == [(pytest.approx(1.2), pytest.approx(0.3))]


def test_ceiling_is_below_full_torque():
    """The whole point: never 100%."""
    assert 0.0 < CEILING < 1.0
