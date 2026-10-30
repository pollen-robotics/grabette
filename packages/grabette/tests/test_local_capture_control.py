"""LAN control must opt out of fleet effects and never stop another episode."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from grabette.models import CaptureStatus


class Backend:
    hardware_error = ""
    busy_reason = ""

    def __init__(self, events):
        self.events = events
        self.capture = CaptureStatus()

    @property
    def is_capturing(self):
        return self.capture.is_capturing

    def get_capture_status(self):
        return self.capture

    async def start_capture(self, directory):
        self.events.append("start")
        self.capture = CaptureStatus(is_capturing=True, episode_id=directory.name)

    async def stop_capture(self):
        self.events.append("stop")
        self.capture.is_capturing = False
        return self.capture


class Scheduler:
    def __init__(self, events):
        self.events = events
        self.pending = False
        self.outcome = "ran"
        self.on_cancel = None

    def is_pending(self):
        return self.pending

    def is_scheduled(self):
        return self.pending

    def mark_waiting_for_fleet(self):
        self.pending = True

    async def cancel_or_wait(self, backend):
        self.events.append("cancel_or_wait")
        if self.on_cancel is not None:
            self.on_cancel()
        if self.outcome in ("cancelled", "cancelled_wait"):
            self.pending = False
        return self.outcome


@pytest.fixture
def rig(tmp_path, monkeypatch):
    # The router owns a module-level TaskManager. Import it only after directing
    # that default constructor to test storage; test collection must never open
    # a workstation's real ~/grabette-data registry.
    monkeypatch.setenv("GRABETTE_DEVICE_ID", "grabette-control-test")
    from grabette.task import TaskManager

    original_init = TaskManager.__init__

    def test_storage(self, data_dir=None):
        return original_init(self, tmp_path / "module-default" if data_dir is None else data_dir)

    with monkeypatch.context() as isolated:
        isolated.setattr(TaskManager, "__init__", test_storage)
        from grabette.app.routers import tasks

    events = []
    backend = Backend(events)
    scheduler = Scheduler(events)
    tm = TaskManager(tmp_path)
    fleet = {"start": {"status": "solo"}, "on_stop": None}

    async def group_start(_):
        events.append("fleet_start")
        return fleet["start"]

    async def group_stop(**_):
        events.append("fleet_stop")
        if fleet["on_stop"] is not None:
            fleet["on_stop"]()

    monkeypatch.setattr(tasks, "get_capture_scheduler", lambda: scheduler)
    monkeypatch.setattr(tasks, "request_group_start", group_start)
    monkeypatch.setattr(tasks, "notify_group_stop", group_stop)
    app = FastAPI()
    app.include_router(tasks.router)
    app.dependency_overrides[tasks.get_backend] = lambda: backend
    app.dependency_overrides[tasks.get_task_manager] = lambda: tm
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, backend=backend, scheduler=scheduler,
                              tm=tm, events=events, fleet=fleet, tasks=tasks)


def start(rig, *, local_only=True):
    response = rig.client.post("/api/episodes/start", json={"local_only": local_only})
    assert response.status_code == 200, response.text
    return response.json()["episode_id"]


def test_control_info_reports_configured_identity(rig, monkeypatch):
    monkeypatch.setattr(rig.tasks, "settings", SimpleNamespace(
        hand="left", device_id="test-left", angle_sensors=True, depth_camera="oakd"))
    assert rig.client.get("/api/capture/control-info").json() == {
        "local_only_control_version": 1, "expected_episode_stop_version": 1,
        "hand": "left", "device_id": "test-left",
        "required_capture_counters": ["frame_count", "angle_sample_count", "imu_sample_count"],
    }


@pytest.mark.parametrize("camera,angles,expected", [
    ("gemini305", True, ["frame_count", "angle_sample_count"]),
    ("stereo_uvc", False, ["frame_count"]),
    ("oakd", False, ["frame_count", "imu_sample_count"]),
])
def test_control_info_only_requires_configured_sensor_counters(rig, monkeypatch, camera, angles, expected):
    monkeypatch.setattr(rig.tasks, "settings", SimpleNamespace(
        hand="right", device_id="test-right", angle_sensors=angles, depth_camera=camera))
    assert rig.client.get("/api/capture/control-info").json()["required_capture_counters"] == expected


def test_local_start_does_not_contact_fleet_even_when_group_would_refuse(rig):
    rig.fleet["start"] = {"status": "refused", "detail": "peer offline"}
    episode = start(rig)
    assert rig.events == ["start"]
    assert rig.backend.capture.episode_id == episode
    assert rig.tm._pending_episode[0] == episode


@pytest.mark.parametrize("body", [None, {}, {"local_only": False}])
def test_default_start_preserves_fleet_behavior(rig, body):
    response = rig.client.post("/api/episodes/start", json=body)
    assert response.status_code == 200
    assert rig.events == ["fleet_start", "start"]


def test_default_start_still_honors_group_refusal(rig):
    rig.fleet["start"] = {"status": "refused", "detail": "peer offline"}
    response = rig.client.post("/api/episodes/start")
    assert response.status_code == 409
    assert rig.events == ["fleet_start"]
    assert rig.tm._pending_episode is None


def test_default_start_can_wait_for_fleet_camera(rig):
    rig.fleet["start"] = {"status": "waiting_camera"}
    response = rig.client.post("/api/episodes/start")
    assert response.json() == {"status": "waiting_camera"}
    assert rig.scheduler.pending
    assert rig.events == ["fleet_start"]


@pytest.mark.parametrize("reason", ["hardware_error", "busy_reason"])
def test_local_start_retains_capture_gate(rig, reason):
    setattr(rig.backend, reason, "not ready")
    response = rig.client.post("/api/episodes/start", json={"local_only": True})
    assert response.status_code == 409
    assert rig.events == []
    assert rig.tm._pending_episode is None


def test_local_stop_is_independent_and_registers_owned_episode(rig):
    episode = start(rig)
    rig.events.clear()
    response = rig.client.post("/api/episodes/stop", json={
        "local_only": True, "expected_episode_id": episode,
    })
    assert response.status_code == 200, response.text
    assert rig.events == ["cancel_or_wait", "stop"]
    assert not rig.backend.is_capturing
    assert rig.tm._pending_episode is None
    assert episode in rig.tm.get_task(rig.tm.active_task_id).episode_ids


@pytest.mark.parametrize("body", [None, {}, {"local_only": False}])
def test_default_stop_preserves_fleet_notification(rig, body):
    start(rig)
    rig.events.clear()
    response = rig.client.post("/api/episodes/stop", json=body)
    assert response.status_code == 200, response.text
    assert rig.events == ["cancel_or_wait", "fleet_stop", "stop"]


def test_mismatched_stop_has_no_scheduler_fleet_or_capture_side_effect(rig):
    episode = start(rig)
    rig.events.clear()
    response = rig.client.post("/api/episodes/stop", json={
        "local_only": True, "expected_episode_id": "another-episode",
    })
    assert response.status_code == 409
    assert rig.events == []
    assert rig.backend.is_capturing
    assert rig.tm._pending_episode[0] == episode


def test_expected_stop_rechecks_identity_after_wait(rig):
    episode = start(rig)
    rig.events.clear()
    rig.scheduler.on_cancel = lambda: setattr(rig.backend.capture, "episode_id", "new-episode")
    response = rig.client.post("/api/episodes/stop", json={
        "local_only": True, "expected_episode_id": episode,
    })
    assert response.status_code == 409
    assert rig.events == ["cancel_or_wait"]
    assert rig.backend.is_capturing


def test_expected_stop_rechecks_identity_after_fleet_notification(rig):
    episode = start(rig)
    rig.events.clear()
    rig.fleet["on_stop"] = lambda: setattr(rig.backend.capture, "episode_id", "new-episode")
    response = rig.client.post("/api/episodes/stop", json={"expected_episode_id": episode})
    assert response.status_code == 409
    assert rig.events == ["cancel_or_wait", "fleet_stop"]
    assert rig.backend.is_capturing


def test_expected_stop_cancels_matching_scheduled_episode(rig):
    episode = rig.tm.create_episode()
    rig.scheduler.pending = True
    rig.scheduler.outcome = "cancelled"
    response = rig.client.post("/api/episodes/stop", json={
        "local_only": True, "expected_episode_id": episode,
    })
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "cancelled"}
    assert rig.events == ["cancel_or_wait"]
    assert rig.tm._pending_episode is None
    assert not rig.tm.episode_dir(episode).exists()


def test_wrong_expected_episode_does_not_cancel_scheduled_start(rig):
    episode = rig.tm.create_episode()
    rig.scheduler.pending = True
    rig.scheduler.outcome = "cancelled"
    response = rig.client.post("/api/episodes/stop", json={
        "local_only": True, "expected_episode_id": "another-episode",
    })
    assert response.status_code == 409
    assert rig.events == []
    assert rig.scheduler.pending
    assert rig.tm._pending_episode[0] == episode


def test_cancel_does_not_discard_a_newly_created_pending_episode(rig):
    old = rig.tm.create_episode(episode_id="20260101_000000")
    new = "20260101_000001"
    rig.scheduler.pending = True
    rig.scheduler.outcome = "cancelled"
    rig.scheduler.on_cancel = lambda: rig.tm.create_episode(episode_id=new)
    response = rig.client.post("/api/episodes/stop", json={
        "local_only": True, "expected_episode_id": old,
    })
    assert response.status_code == 409
    assert rig.tm._pending_episode[0] == new
    assert rig.tm.episode_dir(new).exists()


@pytest.mark.parametrize("local_only", [True, False])
def test_cancelling_fleet_wait_notifies_only_for_default_control(rig, local_only):
    rig.scheduler.pending = True
    rig.scheduler.outcome = "cancelled_wait"
    response = rig.client.post("/api/episodes/stop", json={"local_only": local_only})
    assert response.status_code == 200
    expected = ["cancel_or_wait"] + ([] if local_only else ["fleet_stop"])
    assert rig.events == expected


@pytest.mark.parametrize("route", ["start", "stop"])
@pytest.mark.parametrize("value", ["true", 1, None, []])
def test_local_only_requires_explicit_json_boolean(rig, route, value):
    response = rig.client.post("/api/episodes/" + route, json={"local_only": value})
    assert response.status_code == 422
    assert rig.events == []


def test_empty_expected_episode_is_not_an_unprotected_stop(rig):
    start(rig)
    rig.events.clear()
    response = rig.client.post("/api/episodes/stop", json={"expected_episode_id": ""})
    assert response.status_code == 422
    assert rig.events == []
    assert rig.backend.is_capturing
