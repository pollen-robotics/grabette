"""Workstation control semantics without a recorder, camera, or LAN access."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import threading
import time
import urllib.error

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "grabette_control.py"
spec = importlib.util.spec_from_file_location("dual_control_cli", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

OLD = "20260101_000000"
LATEST = "20260101_000001"
NEW = "20260101_000002"
LEFT = "20260101_000003"
RIGHT = "20260101_000004"


def status(episode=None, count=0, *, active=False, angle=0, imu=0, **extra):
    return {"state": "running", "sensor": {"capture": {
        "episode_id": episode, "is_capturing": active, "is_starting": False,
        "frame_count": count, "angle_sample_count": angle, "imu_sample_count": imu,
        "blocked_reason": "", **extra,
    }}}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeApi:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()
        self.info = {side: {
            "local_only_control_version": 1, "expected_episode_stop_version": 1,
            "hand": side, "device_id": "test-" + side,
            "required_capture_counters": ["frame_count"],
        } for side in ("left", "right")}
        self.states = {side: status() for side in self.info}
        self.ids = {"left": LEFT, "right": RIGHT}
        self.tasks = {side: [{"episode_ids": [OLD, LATEST]}] for side in self.info}
        self.hooks = {}
        self.saved = {}

    def request(self, side, path, body=None, *, method=None):
        method = method or ("GET" if body is None else "POST")
        with self.lock:
            self.calls.append((side, method, path, copy.deepcopy(body)))
        hook = self.hooks.get((side, method, path))
        if hook:
            return hook(body)
        if path == "/api/capture/control-info":
            return copy.deepcopy(self.info[side])
        if path == "/api/daemon/status":
            return copy.deepcopy(self.states[side])
        if path == "/api/episodes/start":
            self.states[side] = status(self.ids[side], active=True)
            return {"episode_id": self.ids[side], "status": "capturing"}
        if path == "/api/episodes/stop":
            cap = self.states[side]["sensor"]["capture"]
            if body["expected_episode_id"] != cap["episode_id"]:
                raise cli.ControlError("Wrong episode", status=409)
            cap["is_capturing"] = False
            return copy.deepcopy(cap)
        if path == "/api/tasks":
            return copy.deepcopy(self.tasks[side])
        if path == "/api/system/info":
            return {"cpu_temp_c": 50}
        if path == "/api/session/status":
            return {"active": False}
        if path.startswith("/api/episodes/"):
            episode = path.rsplit("/", 1)[1]
            if method == "DELETE":
                return {"deleted": episode}
            return copy.deepcopy(self.saved.get((side, episode), {
                "episode_id": episode, "metadata_ok": True, "has_video": True,
                "save_state": "complete",
            }))
        raise AssertionError((side, method, path, body))

    def mutations(self):
        return [call for call in self.calls if call[1] != "GET"]


@pytest.fixture
def rig(tmp_path):
    devices = {side: {"api": "http://" + side + ".example.invalid:8000",
                      "expected_device_id": "test-" + side} for side in ("left", "right")}
    api = FakeApi()
    clock = Clock()
    controller = cli.Controller(devices, cli.EventLog(tmp_path / "logs"), api, clock)
    return controller, api, clock


def fail(message="offline", *, uncertain=False):
    def raise_error(_):
        raise cli.ControlError(message, uncertain=uncertain)
    return raise_error


def test_one_offline_hand_prevents_any_dual_start(rig):
    control, api, _ = rig
    api.hooks["right", "GET", "/api/capture/control-info"] = fail()
    with pytest.raises(cli.ControlError, match="offline"):
        control.start(("left", "right"))
    assert api.mutations() == []


def test_online_hand_can_start_when_other_hand_is_offline(rig):
    control, api, _ = rig
    api.hooks["right", "GET", "/api/capture/control-info"] = fail()
    control.start(("left",))
    assert api.mutations() == [("left", "POST", "/api/episodes/start", {"local_only": True})]


def test_partial_start_rolls_back_only_the_acknowledged_episode(rig):
    control, api, _ = rig
    api.hooks["right", "POST", "/api/episodes/start"] = fail("refused")
    with pytest.raises(cli.ControlError, match="Partial"):
        control.start(("left", "right"))
    stops = [x for x in api.mutations() if x[2].endswith("/stop")]
    assert stops == [("left", "POST", "/api/episodes/stop", {
        "local_only": True, "expected_episode_id": LEFT,
    })]


def test_unknown_start_is_not_retried_or_treated_as_an_owned_failure(rig):
    control, api, _ = rig

    def unknown(_):
        api.states["right"] = status(RIGHT, active=True)
        raise cli.ControlError("lost reply", uncertain=True)

    api.hooks["right", "POST", "/api/episodes/start"] = unknown
    with pytest.raises(cli.ControlError, match="Partial"):
        control.start(("left", "right"))
    assert control.views["right"].start_unknown
    assert len([x for x in api.mutations() if x[:3] == ("right", "POST", "/api/episodes/start")]) == 1
    assert not any(x[:3] == ("right", "POST", "/api/episodes/stop") for x in api.mutations())
    previous = list(api.mutations())
    control.dispatch("start", ("right",))
    assert api.mutations() == previous


def test_rollback_never_stops_a_different_current_episode(rig):
    control, api, _ = rig
    api.states["left"] = status(NEW, active=True)
    assert "skipped" in control.stop("left", owned_episode=LEFT)
    assert api.mutations() == []


@pytest.mark.parametrize("field,value", [
    ("hand", "right"), ("device_id", "another-device"),
    ("local_only_control_version", True), ("expected_episode_stop_version", 0),
])
def test_identity_or_protocol_mismatch_refuses_control(rig, field, value):
    control, api, _ = rig
    api.info["left"][field] = value
    with pytest.raises(cli.ControlError):
        control.start(("left",))
    assert api.mutations() == []


def test_device_change_after_pin_requires_reconnection(rig):
    control, api, _ = rig
    control.views["left"].cfg.pop("expected_device_id")
    control.check_identity("left")
    api.info["left"]["device_id"] = "replacement-left"
    with pytest.raises(cli.ControlError, match="identity changed"):
        control.check_identity("left")
    control.check_identity("left", reconnect=True)


def test_no_imu_camera_can_confirm_main_counter_growth(rig):
    control, api, clock = rig
    control.check_identity("left")
    assert control.views["left"].required == {"frame_count"}
    api.states["left"] = status(LEFT, 1, active=True)
    control.refresh("left")
    assert "waiting" in control.views["left"].phase(clock())
    clock.now = 1
    api.states["left"] = status(LEFT, 31, active=True)
    control.refresh("left")
    assert "growing" in control.views["left"].phase(clock())


def test_required_sensor_must_also_advance_before_confirmation(rig):
    control, api, clock = rig
    api.info["left"]["required_capture_counters"] = ["frame_count", "angle_sample_count"]
    control.check_identity("left")
    for moment, frames, angle in [(0, 1, 0), (1, 30, 0), (2, 60, 10), (3, 90, 20)]:
        clock.now = moment
        api.states["left"] = status(LEFT, frames, angle=angle, active=True)
        control.refresh("left")
        phase = control.views["left"].phase(clock())
        assert ("growing" in phase) is (moment >= 2)


def test_stale_status_never_claims_capture_confirmation():
    view = cli.Recorder({})
    view.observe(status(LEFT, 1, active=True), 1)
    view.observe(status(LEFT, 30, active=True), 2)
    assert "growing" in view.phase(2)
    assert "UNKNOWN" in view.phase(7)


def test_counter_stall_is_visible_while_polling_remains_fresh():
    view = cli.Recorder({})
    for at, count in [(0, 1), (1, 30), (4, 30), (7, 30)]:
        view.observe(status(LEFT, count, active=True), at)
    assert "stalled" in view.phase(7)


def test_never_growing_counter_starting_at_zero_clock_can_stall():
    view = cli.Recorder({})
    for at in (0, 3, 6):
        view.observe(status(LEFT, 1, active=True), at)
    assert "stalled" in view.phase(6)


def test_counter_regression_requires_new_growth_evidence():
    view = cli.Recorder({})
    view.observe(status(LEFT, 10, active=True), 0)
    view.observe(status(LEFT, 40, active=True), 1)
    assert "growing" in view.phase(1)
    view.observe(status(LEFT, 5, active=True), 2)
    assert "waiting" in view.phase(2)
    view.observe(status(LEFT, 35, active=True), 3)
    assert "growing" in view.phase(3)


def test_new_episode_does_not_reuse_previous_episode_confirmation():
    view = cli.Recorder({})
    view.observe(status(LEFT, 10, active=True), 0)
    view.observe(status(LEFT, 40, active=True), 1)
    view.observe(status(RIGHT, 70, active=True), 2)
    assert "waiting" in view.phase(2)


def test_connection_gap_requires_fresh_growth_again():
    view = cli.Recorder({})
    view.observe(status(LEFT, 10, active=True), 0)
    view.observe(status(LEFT, 40, active=True), 1)
    view.error = "lost connection"
    view.observe(status(LEFT, 70, active=True), 2)
    assert "waiting" in view.phase(2)
    view.observe(status(LEFT, 100, active=True), 3)
    assert "growing" in view.phase(3)


def test_capture_without_episode_identity_cannot_be_confirmed():
    view = cli.Recorder({})
    view.observe(status(None, 10, active=True), 0)
    view.observe(status(None, 40, active=True), 1)
    assert "growing" not in view.phase(1)


@pytest.mark.parametrize("value", [-1, float("nan"), None, "10", True])
def test_invalid_required_counter_is_not_recording_evidence(value):
    with pytest.raises(cli.ControlError):
        cli.Recorder({}).observe(status(LEFT, value, active=True), 1)


def test_pending_background_save_does_not_block_a_backend_that_can_capture(rig):
    control, api, _ = rig
    api.states["left"] = status(pending_saves=1, is_saving=True, buffer_bytes=1024)
    control.start(("left",))
    assert any(x[2] == "/api/episodes/start" for x in api.mutations())


def test_backend_blocked_reason_prevents_start_even_without_active_capture(rig):
    control, api, _ = rig
    api.states["left"] = status(blocked_reason="buffer full")
    with pytest.raises(cli.ControlError):
        control.start(("left",))
    assert api.mutations() == []


def test_deletion_keeps_fixed_ids_when_a_new_latest_episode_appears(rig):
    control, api, _ = rig
    control.prepare_delete(("left",))
    assert control.delete_plan["targets"] == {"left": LATEST}
    api.tasks["left"][0]["episode_ids"].append(NEW)
    control.delete()
    assert [x[2] for x in api.mutations()] == ["/api/episodes/" + LATEST]


def test_expired_delete_confirmation_sends_no_delete(rig):
    control, api, clock = rig
    control.prepare_delete(("left",))
    clock.now = 30
    with pytest.raises(cli.ControlError, match="confirmation"):
        control.delete()
    assert api.mutations() == []


def test_delete_rechecks_expiration_after_slow_preflight(rig):
    control, api, clock = rig
    control.prepare_delete(("left",))

    def slow_status(_):
        clock.now = 31
        return status()

    api.hooks["left", "GET", "/api/daemon/status"] = slow_status
    with pytest.raises(cli.ControlError, match="expir|confirmation"):
        control.delete()
    assert api.mutations() == []


def test_all_delete_targets_are_rechecked_before_any_mutation(rig):
    control, api, _ = rig
    control.prepare_delete(("left", "right"))
    api.states["right"] = status(RIGHT, active=True)
    with pytest.raises(cli.ControlError):
        control.delete()
    assert api.mutations() == []


def test_deletion_identity_change_refuses_both_hands(rig):
    control, api, _ = rig
    control.prepare_delete(("left", "right"))
    api.info["right"]["device_id"] = "replacement-right"
    with pytest.raises(cli.ControlError):
        control.delete()
    assert api.mutations() == []


def test_unknown_delete_result_does_not_retry_or_retarget(rig):
    control, api, _ = rig
    control.prepare_delete(("left",))
    path = "/api/episodes/" + LATEST
    api.hooks["left", "DELETE", path] = fail("lost reply", uncertain=True)
    message = control.delete()
    assert "UNKNOWN" in message
    assert api.mutations() == [("left", "DELETE", path, None)]
    assert control.delete_plan is None


def test_active_saving_episode_cannot_be_deleted(rig):
    control, api, _ = rig
    api.saved["left", LATEST] = {"episode_id": LATEST, "save_state": "saving"}
    with pytest.raises(cli.ControlError):
        control.prepare_delete(("left",))
    assert api.mutations() == []


def test_log_write_failure_does_not_prevent_start_or_stop(rig, tmp_path):
    control, api, _ = rig
    blocked = tmp_path / "regular-file"
    blocked.write_text("not a directory")
    control.log = cli.EventLog(blocked)
    control.start(("left",))
    control.stop("left")
    assert len(api.mutations()) == 2
    assert "unavailable" in control.log.error


@pytest.mark.parametrize("method,body", [("POST", {"local_only": True}), ("DELETE", None)])
def test_transport_failure_is_uncertain_and_never_automatically_retried(monkeypatch, method, body):
    calls = []

    class Opener:
        def open(self, request, **kwargs):
            calls.append(request)
            raise TimeoutError("lost reply")

    monkeypatch.setattr(cli.urllib.request, "build_opener", lambda *handlers: Opener())
    api = cli.HttpApi({"left": {"api": "http://left.example.invalid:8000"}})
    with pytest.raises(cli.ControlError) as error:
        api.request("left", "/api/episodes/start", body, method=method)
    assert error.value.uncertain
    assert len(calls) == 1


def test_control_http_redirect_is_not_followed(monkeypatch):
    requests = []
    handlers = []

    class Opener:
        def open(self, request, **kwargs):
            requests.append(request)
            raise urllib.error.HTTPError(request.full_url, 302, "redirect", {}, None)

    def build(*values):
        handlers.extend(values)
        return Opener()

    monkeypatch.setattr(cli.urllib.request, "build_opener", build)
    api = cli.HttpApi({"left": {"api": "http://left.example.invalid:8000"}})
    with pytest.raises(cli.ControlError) as error:
        api.request("left", "/api/episodes/start", {"local_only": True})
    assert error.value.status == 302
    assert len(requests) == 1
    reject = next(x for x in handlers if isinstance(x, cli.NoRedirect))
    assert reject.redirect_request(requests[0], None, 302, "redirect", {}, "http://other.invalid") is None


@pytest.mark.parametrize("path,method,uncertain", [
    ("/api/episodes/start", "POST", True),
    ("/api/daemon/status", "GET", False),
    ("/api/system/info", "GET", False),
    ("/api/episodes/" + LATEST, "GET", False),
])
def test_unexpected_list_reply_is_not_accepted_as_a_state_or_acknowledgement(monkeypatch, path, method, uncertain):
    class Opener:
        def open(self, request, **kwargs):
            return io.StringIO("[]")

    monkeypatch.setattr(cli.urllib.request, "build_opener", lambda *handlers: Opener())
    api = cli.HttpApi({"left": {"api": "http://left.example.invalid:8000"}})
    with pytest.raises(cli.ControlError) as error:
        api.request("left", path, {"local_only": True} if method == "POST" else None)
    assert error.value.uncertain is uncertain


def test_task_list_reply_is_supported(monkeypatch):
    class Opener:
        def open(self, request, **kwargs):
            return io.StringIO('[{"episode_ids": []}]')

    monkeypatch.setattr(cli.urllib.request, "build_opener", lambda *handlers: Opener())
    api = cli.HttpApi({"left": {"api": "http://left.example.invalid:8000"}})
    assert api.request("left", "/api/tasks") == [{"episode_ids": []}]


@pytest.mark.parametrize("url", [
    "http://user:secret@left.example.invalid", "http://:secret@left.example.invalid",
    "http://left.example.invalid/api", "http://left.example.invalid/?token=x",
    "file:///tmp/device", "ftp://left.example.invalid",
])
def test_config_rejects_credentials_or_non_origin_urls(tmp_path, url):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"left": {"api": url}}))
    with pytest.raises(ValueError):
        cli.load_devices(config)


def test_single_hand_configuration_is_supported(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"left": {"api": "http://left.example.invalid:8000"}}))
    assert set(cli.load_devices(config)) == {"left"}


def test_once_snapshot_is_read_only(rig, tmp_path, monkeypatch, capsys):
    _, api, _ = rig
    config = tmp_path / "config.json"
    config.write_text(json.dumps({side: {"api": "http://" + side + ".example.invalid:8000"}
                                  for side in ("left", "right")}))
    monkeypatch.setattr(cli, "HttpApi", lambda devices: api)
    monkeypatch.setattr(cli.sys, "argv", [str(SCRIPT), "--config", str(config), "--once", "--json"])
    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"left", "right"}
    assert api.mutations() == []


@pytest.mark.parametrize("field,value", [("hand", "right"), ("local_only_control_version", 0)])
def test_normal_status_poll_does_not_erase_an_identity_verification_failure(rig, field, value):
    control, api, clock = rig
    original = api.info["left"][field]
    api.info["left"][field] = value
    with pytest.raises(cli.ControlError):
        control.check_identity("left")
    control.poll("left")
    view = control.views["left"]
    assert view.control_error
    assert "UNVERIFIED" in view.phase(clock())
    api.info["left"][field] = original
    control.check_identity("left", reconnect=True)
    assert not view.control_error


def test_late_status_reply_cannot_replace_a_newer_observation():
    view = cli.Recorder({})
    assert view.observe(status(LEFT, 10, active=True), 10, requested=9)
    assert view.observe(status(LEFT, 40, active=True), 11, requested=10)
    assert view.observe(status(OLD, 100, active=True), 12, requested=8) is False
    assert view.request_started == 10
    assert view.at == 11
    assert view.last_episode == LEFT
    assert cli.capture(view.status)["frame_count"] == 40


def test_start_acknowledgement_invalidates_pre_ack_idle_and_active_replies(rig):
    control, api, clock = rig
    clock.now = 10
    api.states["left"] = status(OLD, 90)

    def started(_):
        clock.now = 10.5
        api.states["left"] = status(LEFT, 1, active=True)
        return {"episode_id": LEFT, "status": "capturing"}

    api.hooks["left", "POST", "/api/episodes/start"] = started
    control.start(("left",))
    view = control.views["left"]
    assert view.awaiting_start_observation
    assert view.request_started >= 10.5
    assert "IDLE" not in view.phase(clock())
    assert not view.observe(status(), 10.6, requested=10)
    assert not view.observe(status(OLD, 120, active=True), 10.7, requested=10)
    assert view.last_episode == LEFT
    assert "growing" not in view.phase(10.7)
    clock.now = 11
    control.refresh("left")
    assert "waiting" in view.phase(clock())
    clock.now = 12
    api.states["left"] = status(LEFT, 31, active=True)
    control.refresh("left")
    assert "growing" in view.phase(clock())


def test_stop_can_supersede_inflight_start_without_losing_its_pending_state(rig):
    control, api, _ = rig
    start_entered, release_start = threading.Event(), threading.Event()
    stop_entered, release_stop = threading.Event(), threading.Event()
    stop_saw_idle = threading.Event()

    def start_reply(_):
        start_entered.set()
        assert release_start.wait(3), "test did not release start reply"
        api.states["left"] = status(LEFT, 1, active=True)
        return {"episode_id": LEFT, "status": "capturing"}

    def stop_reply(body):
        assert body == {"local_only": True, "expected_episode_id": LEFT}
        stop_entered.set()
        assert release_stop.wait(3), "test did not release stop reply"
        api.states["left"]["sensor"]["capture"]["is_capturing"] = False
        return copy.deepcopy(api.states["left"]["sensor"]["capture"])

    def daemon_reply(_):
        if not release_start.is_set():
            stop_saw_idle.set()
        return copy.deepcopy(api.states["left"])

    def wait_for(predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            assert time.monotonic() < deadline, "controller worker did not settle"
            time.sleep(.01)

    api.hooks["left", "POST", "/api/episodes/start"] = start_reply
    api.hooks["left", "POST", "/api/episodes/stop"] = stop_reply
    try:
        control.dispatch("start", ("left",))
        assert start_entered.wait(3)
        api.hooks["left", "GET", "/api/daemon/status"] = daemon_reply
        control.dispatch("stop", ("left",))
        stop_token = control.views["left"].pending_token
        assert control.views["left"].pending.startswith("STOP")
        assert stop_saw_idle.wait(3)
        assert control.views["left"].start_inflight
        release_start.set()
        assert stop_entered.wait(3)
        wait_for(lambda: not control.views["left"].start_inflight)
        assert control.views["left"].pending_token == stop_token
        assert control.views["left"].pending.startswith("STOP")
        release_stop.set()
        wait_for(lambda: not control.views["left"].pending)
        assert not cli.capture(api.states["left"])["is_capturing"]
        assert [x[2] for x in api.mutations()] == ["/api/episodes/start", "/api/episodes/stop"]
    finally:
        release_start.set()
        release_stop.set()
        control.shutdown.set()


def test_deleted_episode_is_not_resurrected_by_idle_daemon_state(rig):
    control, api, _ = rig
    api.states["left"] = status(LATEST, 300)
    control.poll("left")
    assert control.views["left"].last_episode == LATEST
    control.prepare_delete(("left",))
    control.delete()
    assert LATEST in control.views["left"].deleted_episodes
    api.calls.clear()
    control.poll("left")
    assert control.views["left"].last_episode is None
    assert control.views["left"].saved == {}
    assert not any(call[2] == "/api/episodes/" + LATEST for call in api.calls)


def test_external_new_episode_cannot_inherit_previous_saved_confirmation(rig):
    control, api, _ = rig
    api.states["left"] = status(OLD, 300)
    control.poll("left")
    assert "SAVED" in control.views["left"].phase(0)
    api.states["left"] = status(NEW, 10)
    api.hooks["left", "GET", "/api/episodes/" + NEW] = fail("new episode metadata unavailable")
    control.poll("left")
    view = control.views["left"]
    assert view.last_episode == NEW
    assert view.saved == {}
    assert "metadata unavailable" in view.save_error
    assert "SAVED" not in view.phase(0)


def test_saved_metadata_for_another_episode_never_claims_current_episode_saved():
    view = cli.Recorder({})
    view.observe(status(NEW, 300), 1)
    view.saved = {"episode_id": OLD, "save_state": "complete", "metadata_ok": True, "has_video": True}
    assert "SAVED" not in view.phase(1)
