"""The dashboard's Open/Close buttons: a gentle ramp to the limits, one move at
a time. The gRPC stub is faked; the real move_gripper() runs."""

import pytest

from gripette import webui
from gripette.config import settings
from gripette.proto import gripper_pb2


class FakeStub:
    """Records every command; motors read `pos`."""

    def __init__(self, pos=(0.0, 0.0)):
        self.pos = pos
        self.goals = []
        self.torque = []

    def ReadMotors(self, req, timeout=None):
        return gripper_pb2.MotorState(motor1_position=self.pos[0], motor2_position=self.pos[1])

    def SetTorque(self, req, timeout=None):
        self.torque.append(req.enable)
        return gripper_pb2.TorqueResponse(success=True)

    def SendMotorCommand(self, req, timeout=None):
        self.goals.append((req.motor1_goal, req.motor2_goal,
                           req.motor1_torque_limit, req.motor2_torque_limit))
        return gripper_pb2.MotorCommandResponse(success=True)


@pytest.fixture
def fake(monkeypatch):
    stub = FakeStub()
    monkeypatch.setattr(webui, "stub", lambda: stub)
    monkeypatch.setattr(webui.time, "sleep", lambda s: None)
    return stub


def test_close_ramps_to_the_limits_with_torque_on(fake):
    code, body = webui.move_gripper("close")
    assert (code, body["status"]) == (200, "done")
    assert fake.torque == [True]
    assert len(fake.goals) == webui.RAMP_STEPS
    # float32 on the wire: compare loosely.
    assert fake.goals[-1][:2] == pytest.approx((settings.motor1_max, settings.motor2_max), abs=1e-6)
    # Monotonic ramp, never past the limits, gentle torque on every step.
    m1 = [g[0] for g in fake.goals]
    assert m1 == sorted(m1) and m1[0] < settings.motor1_max / 2
    assert all(g[2] == g[3] == pytest.approx(webui.GRIPPER_TORQUE) for g in fake.goals)


def test_open_starts_from_a_reading_below_zero_without_going_out_of_range(fake):
    fake.pos = (-0.009, 1.2)
    code, _ = webui.move_gripper("open")
    assert code == 200
    assert all(g[0] >= settings.motor1_min and g[1] >= settings.motor2_min for g in fake.goals)
    assert fake.goals[-1][:2] == (0.0, 0.0)


def test_unknown_action_is_404_and_sends_nothing(fake):
    assert webui.move_gripper("wiggle")[0] == 404
    assert fake.goals == [] and fake.torque == []


def test_a_second_move_while_one_runs_is_refused(fake):
    webui._move_lock.acquire()
    try:
        code, body = webui.move_gripper("close")
    finally:
        webui._move_lock.release()
    assert (code, body["status"]) == (409, "busy")
    assert fake.goals == []
