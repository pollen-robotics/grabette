"""Regression tests for stale-bond deadlock recovery in the BLE WiFi service.

Pins the detection rules both ways: a false negative leaves a robot
unprovisionable (this service is its only route onto WiFi), a false positive
purges a healthy bond. No hardware — dbus/gi are stubbed, being Pi-only
system packages absent in CI.
"""
import sys
import time
import types

import pytest


def _install_dbus_stubs():
    """Put minimal dbus/gi stand-ins in sys.modules, before the import below.

    Only what bluetooth_service touches at import time; the rest is
    monkeypatched per-test.
    """
    if "dbus" in sys.modules:  # real dbus present (e.g. running on a Pi)
        return

    def _passthrough_decorator(*_args, **_kwargs):
        return lambda func: func

    dbus_service = types.ModuleType("dbus.service")
    dbus_service.Object = type("Object", (), {"__init__": lambda self, *a, **k: None})
    dbus_service.method = _passthrough_decorator
    dbus_service.signal = _passthrough_decorator

    class _DBusException(Exception):
        pass

    dbus_exceptions = types.ModuleType("dbus.exceptions")
    dbus_exceptions.DBusException = _DBusException

    dbus_glib = types.ModuleType("dbus.mainloop.glib")
    dbus_glib.DBusGMainLoop = lambda **_kwargs: None
    dbus_mainloop = types.ModuleType("dbus.mainloop")
    dbus_mainloop.glib = dbus_glib

    dbus = types.ModuleType("dbus")
    dbus.service = dbus_service
    dbus.exceptions = dbus_exceptions
    dbus.mainloop = dbus_mainloop
    dbus.DBusException = _DBusException
    dbus.Interface = lambda *a, **k: None
    for name in ("Boolean", "String", "UInt32", "Byte", "Array", "Dictionary",
                 "ObjectPath"):
        setattr(dbus, name, lambda value=None, *a, **k: value)

    glib = types.ModuleType("gi.repository.GLib")
    glib.timeout_add_seconds = lambda *a, **k: 0
    glib.idle_add = lambda *a, **k: 0
    glib.MainLoop = lambda: None
    gi_repository = types.ModuleType("gi.repository")
    gi_repository.GLib = glib
    gi = types.ModuleType("gi")
    gi.repository = gi_repository

    sys.modules.update({
        "dbus": dbus,
        "dbus.service": dbus_service,
        "dbus.exceptions": dbus_exceptions,
        "dbus.mainloop": dbus_mainloop,
        "dbus.mainloop.glib": dbus_glib,
        "gi": gi,
        "gi.repository": gi_repository,
        "gi.repository.GLib": glib,
    })


_install_dbus_stubs()

from grabette.bluetooth import bluetooth_service as bts  # noqa: E402

PEER = "/org/bluez/hci0/dev_50_5A_65_1B_17_EE"
OTHER_PEER = "/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF"


class _RemoveDeviceRecorder:
    """Stands in for the org.bluez.Adapter1 proxy, logging RemoveDevice calls."""

    def __init__(self, raises=None):
        self.removed = []
        self._raises = raises

    def RemoveDevice(self, path):  # noqa: N802 — mirrors the BlueZ method name
        self.removed.append(path)
        if self._raises:
            raise self._raises


@pytest.fixture
def service(monkeypatch):
    """Service with RemoveDevice calls recorded on `service.recorder`.

    `_start_advertising` is neutralised so disconnects don't shell out to btmgmt.
    """
    svc = bts.BluetoothWifiService(device_name="Grabette", pin_code="12345")
    recorder = _RemoveDeviceRecorder()
    svc._adapter = object()
    monkeypatch.setattr(bts.dbus, "Interface", lambda *a, **k: recorder)
    monkeypatch.setattr(svc, "_start_advertising", lambda: None)
    svc.recorder = recorder
    return svc


def _drop(svc, peer=PEER, held=0.1, saw_traffic=False):
    """Simulate one connect→disconnect cycle that lasted `held` seconds."""
    svc._conn_started_at = time.monotonic() - held
    svc._saw_gatt_traffic = saw_traffic
    svc._check_bond_deadlock(peer)


# ---- the deadlock itself ----

def test_purges_bond_after_three_short_silent_drops(service):
    # The field signature (lgrabette-03): connect, ~350ms, drop, repeat.
    for _ in range(3):
        _drop(service)
    assert service.recorder.removed == [PEER]


def test_no_purge_before_the_third_strike(service):
    # Two flaps are ordinary RF flakiness; purging would desync a HEALTHY bond,
    # creating the very loop we're fixing.
    for _ in range(2):
        _drop(service)
    assert service.recorder.removed == []


def test_strike_counter_resets_after_a_purge(service):
    for _ in range(3):
        _drop(service)
    for _ in range(2):
        _drop(service)
    # A second purge must earn its own three strikes, not ride on the first set.
    assert service.recorder.removed == [PEER]
    _drop(service)
    assert service.recorder.removed == [PEER, PEER]


# ---- what must NOT trigger it ----

def test_gatt_traffic_clears_the_strikes(service):
    # A link that carried a command was usable, however fast it then dropped.
    _drop(service)
    _drop(service)
    _drop(service, saw_traffic=True)
    for _ in range(2):
        _drop(service)
    assert service.recorder.removed == []


def test_long_connection_clears_the_strikes(service):
    _drop(service)
    _drop(service)
    _drop(service, held=bts.DEADLOCK_MAX_CONN_SECONDS + 1)
    for _ in range(2):
        _drop(service)
    assert service.recorder.removed == []


def test_strikes_must_come_from_one_peer(service):
    # Two centrals flapping once each is not one stuck bond.
    _drop(service, peer=PEER)
    _drop(service, peer=OTHER_PEER)
    _drop(service, peer=PEER)
    assert service.recorder.removed == []


# ---- robustness of the purge itself ----

def test_purge_survives_a_dbus_error(service, monkeypatch):
    # BlueZ may have pruned the device already; killing the mainloop thread
    # here would stop re-advertising and the Pi would go dark.
    recorder = _RemoveDeviceRecorder(raises=bts.dbus.DBusException("does not exist"))
    monkeypatch.setattr(bts.dbus, "Interface", lambda *a, **k: recorder)
    for _ in range(3):
        _drop(service)
    assert recorder.removed == [PEER]


def test_purge_is_a_noop_without_an_adapter(service):
    service._adapter = None
    for _ in range(3):
        _drop(service)
    assert service.recorder.removed == []


# ---- the wiring (where a regression would actually hide) ----

def test_command_marks_the_link_as_used(service):
    # PING needs no auth and is what the web client opens with.
    service._saw_gatt_traffic = False
    assert service._handle_command(b"PING") == "PONG"
    assert service._saw_gatt_traffic is True


def test_connect_disconnect_signals_drive_the_detector(service):
    # Real BlueZ signal shape, so reworking the PropertiesChanged handling
    # can't silently orphan the detector.
    for _ in range(3):
        service._on_device_properties_changed(
            "org.bluez.Device1", {"Connected": True}, [], path=PEER
        )
        assert service._saw_gatt_traffic is False
        service._on_device_properties_changed(
            "org.bluez.Device1", {"Connected": False}, [], path=PEER
        )
    assert service.recorder.removed == [PEER]


def test_unrelated_properties_are_ignored(service):
    service._on_device_properties_changed(
        "org.bluez.Device1", {"ServicesResolved": True}, [], path=PEER
    )
    service._on_device_properties_changed(
        "org.bluez.Adapter1", {"Connected": False}, [], path=PEER
    )
    assert service.recorder.removed == []
