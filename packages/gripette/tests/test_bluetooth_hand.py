"""HAND command — set this gripette's hand (left/right) over BLE.

The hand is nothing but a pair of motor signs (config.Settings derives
motor1_sign/motor2_sign from it), resolved at service start by
/usr/local/bin/hand-from-hostname, which reads the HOSTNAME and appends
GRIPPER_HAND to /etc/gripette/env. Raspberry Pi Imager 2.0 removed OS
customisation for custom images, so the hostname can no longer be set when
flashing GripetteOS — this command is the replacement.

Setting the hand therefore means three things, in order: set the hostname,
drop the cached GRIPPER_HAND line so the derivation runs again, restart the
unit. The env file is edited line-wise and never rewritten wholesale: the
same file carries per-device calibration appended by other tools.

bluetooth_service imports dbus and gi — system packages that are absent off
a Pi (and in CI) — so they are stubbed before importing it.
"""
import json
import subprocess
import sys
import types

import pytest


def _install_dbus_stubs():
    """Minimal dbus/gi stand-ins, just enough for the module to import.

    setdefault, not assignment: on a real Pi the genuine modules win and
    these tests exercise the real import path.
    """
    dbus = types.ModuleType("dbus")

    class _Object:
        def __init__(self, *args, **kwargs):
            pass

    def _decorator_factory(*args, **kwargs):
        def decorate(fn):
            return fn
        return decorate

    service = types.ModuleType("dbus.service")
    service.Object = _Object
    service.method = _decorator_factory
    service.signal = _decorator_factory
    service.BusName = lambda *a, **k: None

    exceptions = types.ModuleType("dbus.exceptions")

    class DBusException(Exception):
        pass

    exceptions.DBusException = DBusException

    mainloop = types.ModuleType("dbus.mainloop")
    glib = types.ModuleType("dbus.mainloop.glib")
    glib.DBusGMainLoop = lambda *a, **k: None
    mainloop.glib = glib

    dbus.service = service
    dbus.exceptions = exceptions
    dbus.mainloop = mainloop
    dbus.DBusException = DBusException
    dbus.SystemBus = lambda *a, **k: None
    dbus.Interface = lambda *a, **k: None
    dbus.PROPERTIES_IFACE = "org.freedesktop.DBus.Properties"
    dbus.String = str
    dbus.Array = list
    dbus.Dictionary = dict
    dbus.Byte = int
    dbus.Boolean = bool
    dbus.UInt16 = int
    dbus.ObjectPath = str

    gi = types.ModuleType("gi")
    repository = types.ModuleType("gi.repository")
    repository.GLib = types.SimpleNamespace(
        MainLoop=lambda *a, **k: None,
        timeout_add_seconds=lambda *a, **k: None,
    )
    gi.repository = repository
    gi.require_version = lambda *a, **k: None

    for name, module in [
        ("dbus", dbus),
        ("dbus.service", service),
        ("dbus.exceptions", exceptions),
        ("dbus.mainloop", mainloop),
        ("dbus.mainloop.glib", glib),
        ("gi", gi),
        ("gi.repository", repository),
    ]:
        sys.modules.setdefault(name, module)


_install_dbus_stubs()

from gripette.bluetooth import bluetooth_service as bts  # noqa: E402

PIN = "12345"

# An env file as a calibrated device has it: the hand line plus a motor
# offset that MUST survive, since both live in /etc/gripette/env.
ENV_BEFORE = """\
# Gripette persistent configuration.
GRIPPER_HAND=right
GRIPPER_MOTOR1_OFFSET=0.4712
"""


class _Runner:
    """Records subprocess argv and reports success, or failure for one program."""

    def __init__(self, fail_program=None):
        self.calls = []
        self.fail_program = fail_program

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        failed = self.fail_program is not None and argv[0] == self.fail_program
        return subprocess.CompletedProcess(
            argv, 1 if failed else 0, "", "boom" if failed else ""
        )

    def argv(self, program):
        """The first call to `program`, or None if it was never invoked."""
        return next((call for call in self.calls if call[0] == program), None)

    @property
    def programs(self):
        return [call[0] for call in self.calls]


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / "env"
    path.write_text(ENV_BEFORE)
    monkeypatch.setattr(bts, "HAND_ENV_FILE", str(path))
    return path


@pytest.fixture
def runner(monkeypatch):
    stub = _Runner()
    monkeypatch.setattr(bts.subprocess, "run", stub)
    return stub


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(bts.socket, "gethostname", lambda: "gripette")
    return bts.BluetoothWifiService(device_name="Gripette", pin_code=PIN)


def _authenticate(service):
    assert service._handle_command(b"PIN_" + PIN.encode()) == "OK: Connected"


# ---- the query tells the client what the device thinks it is ----

def test_hand_query_needs_no_auth_and_reports_the_current_hand(service, env_file):
    # The hostname is already public in the BLE advert, so reading it back
    # leaks nothing and the client can show state before the PIN is entered.
    status = json.loads(service._handle_command(b"HAND"))
    assert status == {"hand": "right", "hostname": "gripette"}


def test_hand_query_reports_null_when_no_hand_has_been_set(service, env_file):
    env_file.write_text("# nothing set yet\n")
    assert json.loads(service._handle_command(b"HAND"))["hand"] is None


def test_hand_query_reports_null_when_the_env_file_is_absent(service, monkeypatch, tmp_path):
    monkeypatch.setattr(bts, "HAND_ENV_FILE", str(tmp_path / "absent"))
    assert json.loads(service._handle_command(b"HAND"))["hand"] is None


# ---- setting the hand ----

def test_set_hand_requires_auth(service, env_file, runner):
    assert service._handle_command(b"HAND left").startswith("ERROR: Not authenticated")
    assert runner.calls == []
    assert env_file.read_text() == ENV_BEFORE


def test_set_hand_sets_the_hostname(service, env_file, runner):
    _authenticate(service)
    assert service._handle_command(b"HAND left") == "OK: Hand set to left (hostname gripette-left)"
    assert runner.argv("hostnamectl") == ["hostnamectl", "set-hostname", "gripette-left"]


def test_set_hand_clears_the_cached_hand_but_keeps_calibration(service, env_file, runner):
    _authenticate(service)
    service._handle_command(b"HAND left")
    remaining = env_file.read_text()
    assert "GRIPPER_HAND=" not in remaining, "hand-from-hostname would skip re-deriving"
    assert "GRIPPER_MOTOR1_OFFSET=0.4712" in remaining
    assert "# Gripette persistent configuration." in remaining


def test_set_hand_restarts_the_unit_so_the_hand_takes_effect(service, env_file, runner):
    # --no-block: the BLE reply must not wait on the unit coming up (which
    # runs ExecStartPre plus a Python start), or the client times out.
    _authenticate(service)
    service._handle_command(b"HAND left")
    assert runner.argv("systemctl") == [
        "systemctl", "restart", "--no-block", "gripette.service",
    ]


def test_set_hand_consumes_the_pin(service, env_file, runner):
    _authenticate(service)
    service._handle_command(b"HAND left")
    assert service._handle_command(b"HAND right").startswith("ERROR: Not authenticated")


def test_set_hand_accepts_either_hand_in_any_case(service, env_file, runner):
    _authenticate(service)
    assert service._handle_command(b"HAND RIGHT").startswith("OK:")
    assert runner.argv("hostnamectl")[-1] == "gripette-right"


# ---- a root service taking BLE input: the value is allowlisted ----

@pytest.mark.parametrize("payload", [
    b"HAND",  # handled as the query, never as a set
    b"HAND middle",
    b"HAND left; rm -rf /",
    b"HAND $(id)",
    b"HAND ../../etc/passwd",
])
def test_set_hand_refuses_anything_but_left_or_right(service, env_file, runner, payload):
    _authenticate(service)
    service._handle_command(payload)
    assert runner.argv("hostnamectl") is None
    assert env_file.read_text() == ENV_BEFORE


def test_set_hand_reports_the_error_for_an_unknown_hand(service, env_file, runner):
    _authenticate(service)
    assert service._handle_command(b"HAND middle") == "ERROR: Hand must be 'left' or 'right'."


# ---- a failed hostname change must not half-apply ----

def test_set_hand_leaves_everything_alone_when_hostnamectl_fails(service, env_file, monkeypatch):
    stub = _Runner(fail_program="hostnamectl")
    monkeypatch.setattr(bts.subprocess, "run", stub)
    _authenticate(service)
    assert service._handle_command(b"HAND left").startswith("ERROR:")
    assert env_file.read_text() == ENV_BEFORE, "hand cleared despite the hostname not changing"
    assert "systemctl" not in stub.programs, "restarted into the old hostname"
