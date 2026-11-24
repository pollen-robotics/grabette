"""Servo-side zero re-centring (STS3215 offset register).

Found on hardware: a right gripette calibrated with GRIPPER_MOTOR1_OFFSET=2.908
and GRIPPER_MOTOR2_OFFSET=2.600 stopped closing at 13 deg and 31 deg. The open
pose sat near the encoder's 0/4095 seam (+-pi), so goals offset + angle ran
past it: pi - 2.908 = 13.4 deg, pi - 2.600 = 31.0 deg. Calibration now
re-centres the servo's own zero on the open pose and refuses offsets that
cannot reach the joint limits.

The fake servo below models the register the way the STS3215 does:
present = (actual - offset) mod 4096, offset in sign-magnitude (bit 11).
"""

import math
import sys
import types

import pytest

if "serial" not in sys.modules:
    import importlib.machinery

    _stub = types.ModuleType("serial")
    _stub.__spec__ = importlib.machinery.ModuleSpec("serial", None)
    _stub.Serial = object
    _stub.SerialException = Exception
    sys.modules["serial"] = _stub

from gripette.hardware import motors as m  # noqa: E402
from gripette.hardware.motors import (  # noqa: E402
    MotorController, centering_offset, closing_travel, decode_sts_offset,
    encode_sts_offset, travel_problem)

LIM1 = math.radians(93.5)
LIM2 = math.radians(116.0)


def steps_to_rad(steps: int) -> float:
    return (steps - m.STS_CENTER_STEPS) * 2 * math.pi / m.STS_STEPS_PER_TURN


def rad_to_steps(rad: float) -> int:
    return round(rad * m.STS_STEPS_PER_TURN / (2 * math.pi)) + m.STS_CENTER_STEPS


class FakeSts3215:
    """Two servos, each at a physical angle `actual` (steps, one turn)."""

    def __init__(self, actual, offsets=(0, 0), ids=(1, 2)):
        self.actual = dict(zip(ids, actual))
        self.reg_offset = {i: encode_sts_offset(o) for i, o in zip(ids, offsets)}
        self.locked = {i: 1 for i in ids}
        self.goal = {}
        self.torque = {i: True for i in ids}
        self.offset_writes = 0

    def _present(self, i):
        return (self.actual[i] - decode_sts_offset(self.reg_offset[i])) % 4096

    # --- the rustypot calls MotorController makes ---------------------------
    # Single-servo reads return a list, as rustypot does on hardware (a bare
    # int here hid a TypeError that the first real calibration hit).
    def read_raw_present_position(self, i):
        return [self._present(i)]

    def sync_read_present_position(self, ids):
        return [steps_to_rad(self._present(i)) for i in ids]

    def sync_read_present_load(self, ids):
        return [0 for _ in ids]

    def read_raw_data(self, i, addr, length):
        assert (addr, length) == (m.STS_ADDR_OFFSET, 2)
        raw = self.reg_offset[i]
        return [[raw & 0xFF, raw >> 8]]

    def write_raw_data(self, i, addr, data):
        if addr == m.STS_ADDR_LOCK:
            self.locked[i] = data[0]
        elif addr == m.STS_ADDR_OFFSET:
            assert self.locked[i] == 0, "EEPROM written while locked"
            assert not self.torque[i], "offset written with torque on"
            self.reg_offset[i] = data[0] | (data[1] << 8)
            self.offset_writes += 1
        else:
            raise AssertionError(f"unexpected write at {addr}")

    def write_raw_goal_position(self, i, value):
        self.goal[i] = value

    def sync_write_goal_position(self, ids, values):
        for i, v in zip(ids, values):
            self.goal[i] = rad_to_steps(v)

    def sync_write_torque_enable(self, ids, values):
        for i, v in zip(ids, values):
            self.torque[i] = bool(v)


def controller(fake, offsets=(0.0, 0.0)):
    c = MotorController(limits=((0.0, LIM1), (0.0, LIM2)), offsets=offsets)
    c._mock = False
    c._controller = fake
    return c


# --- register encoding --------------------------------------------------------

@pytest.mark.parametrize("steps", [0, 1, -1, 900, -900, 2047, -2047])
def test_offset_register_round_trips(steps):
    assert decode_sts_offset(encode_sts_offset(steps)) == steps


def test_negative_offsets_use_bit_11_as_sign():
    assert encode_sts_offset(-5) == 0x800 | 5


def test_out_of_range_offset_is_refused():
    with pytest.raises(ValueError):
        encode_sts_offset(2048)


@pytest.mark.parametrize("present,current", [
    (4033, 0), (3943, 0), (10, 0), (2048, 0), (100, -1500), (4000, 1900)])
def test_centering_offset_brings_the_pose_to_mid_range(present, current):
    new = centering_offset(present, current)
    assert -2047 <= new <= 2047
    actual = present + current
    assert abs((actual - new) % 4096 - 2048) <= 1


# --- the field case -----------------------------------------------------------

FIELD_OFFSETS = (2.908428, 2.600097)  # rad, right hand


def test_the_field_offsets_explain_the_13_and_31_degree_stops():
    assert math.degrees(closing_travel(FIELD_OFFSETS[0], +1)) == pytest.approx(13.4, abs=0.1)
    assert math.degrees(closing_travel(FIELD_OFFSETS[1], +1)) == pytest.approx(31.0, abs=0.1)


def test_the_field_offsets_are_refused_by_calibration():
    problem = travel_problem(FIELD_OFFSETS, (+1, +1), (LIM1, LIM2))
    assert problem and "motor 1" in problem and "motor 2" in problem


def test_a_centred_calibration_passes_on_either_hand():
    assert travel_problem((0.01, -0.02), (+1, +1), (LIM1, LIM2)) is None
    assert travel_problem((0.01, -0.02), (-1, -1), (LIM1, LIM2)) is None


def test_left_hand_runs_out_on_the_other_side():
    """A left hand closes towards negative encoder values: its seam is at -pi."""
    assert travel_problem((2.9, 2.6), (-1, -1), (LIM1, LIM2)) is None
    assert travel_problem((-2.9, -2.6), (-1, -1), (LIM1, LIM2)) is not None


def test_recentring_the_field_gripette_restores_the_full_travel():
    open_pose = [rad_to_steps(o) for o in FIELD_OFFSETS]
    fake = FakeSts3215(open_pose)
    c = controller(fake, offsets=FIELD_OFFSETS)

    r = c._do_center()

    assert r["changed"]
    assert all(abs(p - 2048) <= 1 for p in r["present"])
    assert all(fake.locked[i] == 1 for i in (1, 2)), "EEPROM left unlocked"
    assert not any(fake.torque.values())
    # The goal register is parked on the new reading, not left in the old frame.
    assert fake.goal == {1: r["present"][0], 2: r["present"][1]}

    # The existing calibration math, run on the cache right after: the new
    # software offset is the residual and leaves the full travel.
    robot = c.read_positions()
    new = [old + rb * s for old, rb, s in zip(FIELD_OFFSETS, robot, (1, 1))]
    assert all(abs(o) < 0.01 for o in new)
    assert travel_problem(new, (1, 1), (LIM1, LIM2)) is None


def test_goals_and_torque_on_are_refused_until_restart():
    fake = FakeSts3215([rad_to_steps(o) for o in FIELD_OFFSETS])
    c = controller(fake, offsets=FIELD_OFFSETS)
    c._do_center()
    with pytest.raises(RuntimeError, match="re-centred"):
        c.write_goal_positions(0.1, 0.1)
    with pytest.raises(RuntimeError, match="re-centred"):
        c.set_torque(True)
    c.set_torque(False)  # releasing stays possible


def test_an_already_centred_servo_is_not_rewritten():
    fake = FakeSts3215([2050, 2040])
    c = controller(fake)
    r = c._do_center()
    assert not r["changed"]
    assert fake.offset_writes == 0
    c.write_goal_positions(0.1, 0.1)  # frame unchanged: goals still accepted


def test_recentring_is_idempotent_with_an_existing_servo_offset():
    fake = FakeSts3215([rad_to_steps(2.9), rad_to_steps(2.6)], offsets=(-300, 700))
    c = controller(fake)
    c._do_center()
    first = dict(fake.reg_offset)
    c._do_center()
    assert fake.reg_offset == first
    assert all(abs(fake._present(i) - 2048) <= 1 for i in (1, 2))


def test_a_failed_check_restores_the_old_offset():
    """If the reading after the write is not mid-range (gripper moved, or a
    servo that interprets the register differently), roll back."""

    class MovingFake(FakeSts3215):
        def write_raw_data(self, i, addr, data):
            super().write_raw_data(i, addr, data)
            if addr == m.STS_ADDR_OFFSET:
                self.actual[i] += 200  # the finger moved during the write

    fake = MovingFake([rad_to_steps(2.9), 2048], offsets=(5, 0))
    c = controller(fake)
    with pytest.raises(RuntimeError, match="offset restored"):
        c._do_center()
    assert decode_sts_offset(fake.reg_offset[1]) == 5
    assert fake.locked[1] == 1


def test_center_servo_offsets_runs_on_the_bus_thread():
    """The public call must go through the bus owner, never touch the bus
    from the caller's thread."""
    fake = FakeSts3215([rad_to_steps(o) for o in FIELD_OFFSETS])
    c = controller(fake, offsets=FIELD_OFFSETS)
    c._running = True
    import threading
    c._thread = threading.Thread(target=c._bus_loop, daemon=True)
    c._thread.start()
    try:
        r = c.center_servo_offsets(timeout=2.0)
    finally:
        c.stop()
    assert r["changed"]
    assert all(abs(fake._present(i) - 2048) <= 1 for i in (1, 2))
