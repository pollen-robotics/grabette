"""Regression tests for stale-bond deadlock recovery in the BLE WiFi service.

Pins the detection rules both ways: a false negative leaves a robot
unprovisionable (this service is its only route onto WiFi), a false positive
purges a healthy bond. No hardware — dbus/gi are stubbed, being Pi-only
system packages absent in CI.
"""
import time
import types

import pytest

from grabette.bluetooth import bluetooth_service as bts

PEER = "/org/bluez/hci0/dev_50_5A_65_1B_17_EE"
OTHER_PEER = "/org/bluez/hci0/dev_AA_BB_CC_DD_EE_FF"


class _BlueZStub:
    """Stands in for both BlueZ proxies the detector uses.

    Adapter1.RemoveDevice (logged) and Properties.Get, which answers the
    Bonded lookup gating a strike. Bonded defaults True: these tests model a
    real stale bond, which by definition is one we hold keys for.
    """

    def __init__(self, raises=None, bonded=True):
        self.removed = []
        self.bonded = bonded
        self._raises = raises

    def RemoveDevice(self, path):  # noqa: N802 — mirrors the BlueZ method name
        self.removed.append(path)
        if self._raises:
            raise self._raises

    def Get(self, interface, prop):  # noqa: N802 — mirrors the DBus method name
        return self.bonded


@pytest.fixture
def service(monkeypatch):
    """Service with RemoveDevice calls recorded on `service.recorder`.

    `_start_advertising` is neutralised so disconnects don't shell out to btmgmt.
    """
    svc = bts.BluetoothWifiService(device_name="Grabette", pin_code="12345")
    recorder = _BlueZStub()
    svc._adapter = object()
    # Only get_object is reached, and its result goes straight to the patched
    # dbus.Interface below.
    svc.bus = types.SimpleNamespace(get_object=lambda *a, **k: None)
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


def test_no_strike_when_we_hold_no_bond(service):
    # Seen for real on grabette-simsim: a probe (bluetoothctl connect, or a
    # chooser being closed) holds the link ~1s and leaves, and the detector
    # counted strikes against it. Without keys of our own there is also
    # nothing to purge — dropping OUR key is the entire mechanism.
    service.recorder.bonded = False
    for _ in range(5):
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
    recorder = _BlueZStub(raises=bts.dbus.DBusException("does not exist"))
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


# ---- the first-connection edge (BlueZ announces it on a different signal) ----

def test_first_time_central_is_tracked_via_interfaces_added(service):
    # A never-seen central appears as InterfacesAdded carrying Connected=true,
    # NOT as PropertiesChanged. Missing it leaves _conn_started_at at 0, so
    # every later drop measures as uptime-long and the detector never fires.
    service._on_interfaces_added(PEER, {"org.bluez.Device1": {"Connected": True}})
    assert service._connected_device_path == PEER
    assert service._conn_started_at > 0
    assert service._saw_gatt_traffic is False


def test_interfaces_added_ignores_the_irrelevant(service):
    service._on_interfaces_added(PEER, {"org.bluez.Device1": {"Connected": False}})
    service._on_interfaces_added("/org/bluez/hci0", {"org.bluez.Adapter1": {}})
    assert service._connected_device_path is None


def test_detector_fires_when_the_connect_edge_came_from_interfaces_added(service):
    # The real first-contact sequence: InterfacesAdded to connect,
    # PropertiesChanged to drop.
    for _ in range(3):
        service._on_interfaces_added(PEER, {"org.bluez.Device1": {"Connected": True}})
        service._on_device_properties_changed(
            "org.bluez.Device1", {"Connected": False}, [], path=PEER
        )
    assert service.recorder.removed == [PEER]


def test_service_discovery_marks_the_link_as_used(service):
    # Service discovery runs over ATT, so it can only happen past SMP.
    service._saw_gatt_traffic = False
    service._on_device_properties_changed(
        "org.bluez.Device1", {"ServicesResolved": True}, [], path=PEER
    )
    assert service._saw_gatt_traffic is True


def test_a_short_but_working_session_is_not_a_stale_bond(service):
    # Bonded, well under DEADLOCK_MAX_CONN_SECONDS, and no command written —
    # but it discovered services, which a bond dying in SMP never reaches.
    for _ in range(3):
        service._on_central_connected(PEER)
        service._on_device_properties_changed(
            "org.bluez.Device1", {"ServicesResolved": True}, [], path=PEER
        )
        service._on_device_properties_changed(
            "org.bluez.Device1", {"Connected": False}, [], path=PEER
        )
    assert service.recorder.removed == []
