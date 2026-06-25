"""Bluetooth LE WiFi configuration service for Gripette.

Exposes a BLE GATT service that allows configuring WiFi credentials
from a phone or laptop (via Web Bluetooth or any BLE client).

Adapted from reachy_mini bluetooth_service.py:
https://github.com/pollen-robotics/reachy_mini/tree/main/src/reachy_mini/daemon/app/services/bluetooth

Changes from reference:
- Device name: "Gripette"
- PIN: from env var GRIPPER_BT_PIN (not dfu-util serial)
- Direct nmcli (no CMD_ shell scripts); crypto done in-process (no daemon split)
- No Device Info Service (unnecessary)
- Simplified status service (network status only)

The WiFi password is never sent in clear over the BLE link. The client runs
WIFI_KEYEX to fetch the robot's ephemeral X25519 public key, derives a shared
key with HKDF-SHA256 (salt = PIN), and seals the password with AES-256-GCM
(AAD = SSID); the sealed blob is sent via WIFI_CONNECT_ENC. This mirrors the
reachy_mini scheme, with crypto done in-process here rather than in a daemon.
"""

import base64
import json
import logging
import os
import socket
import subprocess
import threading
import time
from typing import Callable

import dbus
import dbus.mainloop.glib
import dbus.service
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from gi.repository import GLib

logger = logging.getLogger(__name__)

# ---- BLE UUIDs ----

# Command service: write commands, read responses
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
COMMAND_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"
RESPONSE_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef2"

# Status service: readable network status (auto-updates every 10s)
STATUS_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef3"
NETWORK_STATUS_UUID = "12345678-1234-5678-1234-56789abcdef4"

# ---- BlueZ DBus constants ----

BLUEZ_SERVICE_NAME = "org.bluez"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"
GATT_DESC_IFACE = "org.bluez.GattDescriptor1"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
AGENT_PATH = "/org/bluez/agent"

# Descriptor UUIDs
USER_DESCRIPTION_UUID = "00002901-0000-1000-8000-00805f9b34fb"

# ---- WiFi password sealing (X25519 + HKDF-SHA256 + AES-256-GCM) ----

# HKDF context string — must match the web client byte-for-byte. Shared by all
# robot types (Grabette/Gripette/Casquette) on purpose: the web client offers a
# single device chooser and seals against whichever robot is selected, so a
# per-robot string would break cross-provisioning. It is only domain separation
# from other protocols — secrecy comes from the per-session X25519 keys, not this.
HKDF_INFO = b"grabette-wifi-psk-v1"
# Algorithm tag advertised in the WIFI_KEYEX reply.
KEYEX_ALG = "x25519-hkdf-sha256-aesgcm"

# ---- PIN brute-force rate limiting ----

# A PIN is short (the default is 5 digits), so without a limiter a central in
# BLE range could exhaust the keyspace in minutes by spamming PIN_xxxxx. After
# MAX_PIN_ATTEMPTS consecutive wrong PINs we refuse further attempts for a
# lockout window that DOUBLES on each repeated lockout (PIN_LOCKOUT_SECONDS,
# 2×, 4×, … capped at MAX_PIN_LOCKOUT_SECONDS), so a determined guesser is
# slowed geometrically. The counters live on the service, not the connection,
# and are NOT reset on disconnect — an attacker can't clear them by dropping and
# re-opening the BLE link. Only a correct PIN (or a service restart) resets them.
MAX_PIN_ATTEMPTS = 5
PIN_LOCKOUT_SECONDS = 30
MAX_PIN_LOCKOUT_SECONDS = 3600


# =====================================================================
# BLE Agent — "Just Works" pairing (no user interaction on device side)
# =====================================================================

class NoInputAgent(dbus.service.Object):
    """BLE Agent for Just Works pairing (NoInputNoOutput capability)."""

    # NB: the method signatures below MUST match the org.bluez.Agent1 API
    # exactly. BlueZ invokes them with arguments (device object path, passkey,
    # uuid, …); declaring in_signature="" desyncs the introspected signature
    # from the call and makes BlueZ cancel the pairing — which manifests
    # client-side as "Connection attempt failed", especially with laptops that
    # negotiate numeric-comparison instead of plain Just Works.

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Release(self):
        logger.info("Agent released")

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="s")
    def RequestPinCode(self, device):
        logger.info("RequestPinCode (%s) — returning empty (Just Works)", device)
        return ""

    @dbus.service.method("org.bluez.Agent1", in_signature="os", out_signature="")
    def DisplayPinCode(self, device, pincode):
        pass

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="u")
    def RequestPasskey(self, device):
        logger.info("RequestPasskey (%s) — returning 0 (Just Works)", device)
        return dbus.UInt32(0)

    @dbus.service.method("org.bluez.Agent1", in_signature="ouq", out_signature="")
    def DisplayPasskey(self, device, passkey, entered):
        pass

    @dbus.service.method("org.bluez.Agent1", in_signature="ou", out_signature="")
    def RequestConfirmation(self, device, passkey):
        logger.info("RequestConfirmation (%s) — auto-accepting", device)

    @dbus.service.method("org.bluez.Agent1", in_signature="o", out_signature="")
    def RequestAuthorization(self, device):
        logger.info("RequestAuthorization (%s) — auto-accepting", device)

    @dbus.service.method("org.bluez.Agent1", in_signature="os", out_signature="")
    def AuthorizeService(self, device, uuid):
        logger.info("AuthorizeService (%s, %s) — auto-accepting", device, uuid)

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Cancel(self):
        logger.info("Agent request canceled")


# =====================================================================
# GATT base classes: Descriptor, Characteristic, Service
# =====================================================================

class Descriptor(dbus.service.Object):
    """GATT Descriptor."""

    def __init__(self, bus, index, uuid, flags, characteristic):
        self.path = characteristic.path + "/desc" + str(index)
        self.bus = bus
        self.uuid = uuid
        self.flags = flags
        self.characteristic = characteristic
        self.value = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_DESC_IFACE: {
                "Characteristic": self.characteristic.get_path(),
                "UUID": self.uuid,
                "Flags": self.flags,
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != GATT_DESC_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs", "Unknown interface"
            )
        return self.get_properties()[GATT_DESC_IFACE]

    @dbus.service.method(GATT_DESC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options):
        return dbus.Array(self.value, signature="y")

    @dbus.service.method(GATT_DESC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value, options):
        self.value = value


class Characteristic(dbus.service.Object):
    """GATT Characteristic base class."""

    def __init__(self, bus, index, uuid, flags, service):
        self.path = service.path + "/char" + str(index)
        self.bus = bus
        self.uuid = uuid
        self.service = service
        self.flags = flags
        self.value = []
        self.descriptors = []
        dbus.service.Object.__init__(self, bus, self.path)

    @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
    def PropertiesChanged(self, interface, changed, invalidated):
        pass

    def get_properties(self):
        props = {
            GATT_CHRC_IFACE: {
                "Service": self.service.get_path(),
                "UUID": self.uuid,
                "Flags": self.flags,
            }
        }
        if self.descriptors:
            props[GATT_CHRC_IFACE]["Descriptors"] = [
                d.get_path() for d in self.descriptors
            ]
        return props

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != GATT_CHRC_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs", "Unknown interface"
            )
        return self.get_properties()[GATT_CHRC_IFACE]

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options):
        return dbus.Array(self.value, signature="y")

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value, options):
        self.value = value


# ---- Specialized characteristics ----

class CommandCharacteristic(Characteristic):
    """Write-only characteristic that dispatches commands to a handler."""

    def __init__(self, bus, index, service, command_handler: Callable[[bytes], str]):
        super().__init__(bus, index, COMMAND_CHAR_UUID, ["write"], service)
        self.command_handler = command_handler

    def WriteValue(self, value, options):
        command_bytes = bytes(value)
        # Run in a background thread so the GATT write returns immediately.
        # Commands like WIFI can block for several seconds (nmcli), which would
        # cause the BLE client to time out with "GATT operation failed".
        threading.Thread(target=self._run_command, args=(command_bytes,), daemon=True).start()

    def _run_command(self, command_bytes: bytes) -> None:
        response = self.command_handler(command_bytes)
        # Deliver the result on the GLib main loop thread (DBus is not
        # thread-safe). The client awaits exactly one notification per command,
        # so we emit a single final response (never an intermediate ack).
        def _update():
            self.service.response_char.send_notification(response)
            logger.info("Command processed, response: %s", response)
            return False  # one-shot
        GLib.idle_add(_update)


class ResponseCharacteristic(Characteristic):
    """Read/notify characteristic that holds the last command response."""

    def __init__(self, bus, index, service):
        super().__init__(bus, index, RESPONSE_CHAR_UUID, ["read", "notify"], service)

    def send_notification(self, text: str) -> None:
        """Store the result and notify subscribed clients.

        Must run on the GLib mainloop thread (DBus signal emission). The value
        is stored so a reading client still sees it; BlueZ only forwards the
        PropertiesChanged signal to centrals that subscribed.
        """
        encoded = [dbus.Byte(b) for b in text.encode("utf-8")]
        self.value = encoded
        self.PropertiesChanged(
            GATT_CHRC_IFACE,
            {"Value": dbus.Array(encoded, signature="y")},
            dbus.Array([], signature="s"),
        )


class DynamicCharacteristic(Characteristic):
    """Read-only characteristic whose value is refreshed by a callable."""

    def __init__(self, bus, index, uuid, service, value_getter, description=None):
        super().__init__(bus, index, uuid, ["read"], service)
        self.value_getter = value_getter
        self.update_value()
        if description:
            desc = Descriptor(bus, 0, USER_DESCRIPTION_UUID, ["read"], self)
            desc.value = [dbus.Byte(b) for b in description.encode("utf-8")]
            self.add_descriptor(desc)

    def update_value(self):
        """Refresh value from the getter. Returns True to keep GLib timer alive."""
        value_str = self.value_getter()
        self.value = [dbus.Byte(b) for b in value_str.encode("utf-8")]
        return True


# =====================================================================
# GATT Services
# =====================================================================

class CommandService(dbus.service.Object):
    """Primary GATT service with command/response characteristics."""

    PATH_BASE = "/org/bluez/service"

    def __init__(self, bus, index, uuid, command_handler: Callable[[bytes], str]):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.uuid = uuid
        self.primary = True
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)

        # Response first (so command handler can reference it)
        self.response_char = ResponseCharacteristic(bus, 1, self)
        self.characteristics.append(self.response_char)
        self.characteristics.append(CommandCharacteristic(bus, 0, self, command_handler))

    def get_properties(self):
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": self.primary,
                "Characteristics": [ch.get_path() for ch in self.characteristics],
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != GATT_SERVICE_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs", "Unknown interface"
            )
        return self.get_properties()[GATT_SERVICE_IFACE]


class StatusService(dbus.service.Object):
    """GATT service exposing network status (auto-updates every 10s)."""

    PATH_BASE = "/org/bluez/status"

    def __init__(self, bus, index):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.uuid = STATUS_SERVICE_UUID
        self.primary = True
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)

        self.network_char = DynamicCharacteristic(
            bus, 0, NETWORK_STATUS_UUID, self, get_network_status, "Network Status"
        )
        self.characteristics.append(self.network_char)

    def update_network_status(self):
        """Periodic refresh — returns True to keep GLib timer alive."""
        self.network_char.update_value()
        return True

    def get_properties(self):
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": self.primary,
                "Characteristics": [ch.get_path() for ch in self.characteristics],
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != GATT_SERVICE_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs", "Unknown interface"
            )
        return self.get_properties()[GATT_SERVICE_IFACE]


# =====================================================================
# GATT Application (aggregates all services)
# =====================================================================

class Application(dbus.service.Object):
    """GATT Application — container for all BLE services."""

    def __init__(self, bus, command_handler: Callable[[bytes], str]):
        self.path = "/"
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)

        self.services.append(CommandService(bus, 0, SERVICE_UUID, command_handler))
        self.status_service = StatusService(bus, 1)
        self.services.append(self.status_service)

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self):
        resp = {}
        for service in self.services:
            resp[service.get_path()] = service.get_properties()
            for ch in service.characteristics:
                resp[ch.get_path()] = ch.get_properties()
                for desc in ch.descriptors:
                    resp[desc.get_path()] = desc.get_properties()
        return resp


# =====================================================================
# Network helpers
# =====================================================================

def _active_wifi_ssid() -> str:
    """Name of the currently-connected WiFi network, or "" if none.

    Reads the active connections (no scan, so it stays fast enough for the
    periodic mainloop refresh): the active 802-11-wireless connection's NAME is
    the SSID (NetworkManager names WiFi connections after the SSID by default).
    """
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "NAME,TYPE", "connection", "show", "--active"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.endswith(":802-11-wireless"):
                name = line[: -len(":802-11-wireless")]
                return name.replace("\\:", ":")  # un-escape nmcli's ':'
    except Exception:
        pass
    return ""


def get_network_status() -> str:
    """Return network status string: "{MODE} (ssid) [iface] ip ; [iface] ip".

    MODE is one of: CONNECTED, HOTSPOT, OFFLINE. The "(ssid)" segment is present
    only when connected to a WiFi network.
    """
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show"], capture_output=True, text=True
        )
        interfaces = {}
        current_iface = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line.startswith("inet"):
                parts = line.split(":")
                if len(parts) >= 2:
                    iface = parts[1].strip()
                    if iface != "lo":
                        current_iface = iface
            elif line.startswith("inet ") and current_iface:
                ip_addr = line.split()[1].split("/")[0]
                interfaces[current_iface] = ip_addr

        if not interfaces:
            return "OFFLINE"

        wlan_ip = interfaces.get("wlan0", "")
        mode = "HOTSPOT" if wlan_ip.startswith("10.42.0.") else "CONNECTED"
        parts = [f"[{iface}] {ip}" for iface, ip in interfaces.items()]
        ssid = _active_wifi_ssid() if mode == "CONNECTED" else ""
        prefix = f"{mode} ({ssid})" if ssid else mode
        return f"{prefix} {' ; '.join(parts)}"
    except Exception as e:
        logger.error("Error getting network status: %s", e)
        return "ERROR"


# A scan reply must fit a single BLE notification, whose payload is bounded by
# the negotiated ATT MTU. 180 bytes is safe even on a small MTU; SSIDs beyond
# that budget are dropped (strongest-signal networks are kept first).
WIFI_SCAN_MTU_BUDGET = 180


def _wifi_scan() -> str:
    """Rescan and return nearby SSIDs as a JSON array, strongest signal first.

    The list is bounded to WIFI_SCAN_MTU_BUDGET bytes so it fits one BLE
    notification. Returns an "ERROR: ..." string on failure.
    """
    # Best-effort active rescan so the list isn't stale; ignore failures
    # (the radio may be briefly busy) and fall back to the cached list.
    try:
        subprocess.run(
            ["nmcli", "device", "wifi", "rescan"],
            capture_output=True, text=True, timeout=12,
        )
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "SSID,SIGNAL", "device", "wifi", "list"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        return f"ERROR: {e}"

    seen: set = set()
    networks = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        # -t output is "SSID:SIGNAL"; SIGNAL is the numeric last field, and
        # any ':' inside the SSID is backslash-escaped by nmcli.
        ssid, _, signal = line.rpartition(":")
        ssid = ssid.replace("\\:", ":").replace("\\\\", "\\").strip()
        if not ssid or ssid in seen:  # skip hidden (empty) and duplicates
            continue
        seen.add(ssid)
        try:
            strength = int(signal)
        except ValueError:
            strength = 0
        networks.append((strength, ssid))

    networks.sort(key=lambda n: n[0], reverse=True)
    out: list = []
    for _, ssid in networks:
        trial = out + [ssid]
        encoded = json.dumps(trial, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > WIFI_SCAN_MTU_BUDGET:
            break
        out = trial
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def _wifi_connect(ssid: str, password: str) -> str:
    """Connect to a WiFi network using nmcli. Returns status message.

    Robust against a stale scan cache: ``nmcli device wifi connect`` infers the
    AP's security from the scan list, so if the network isn't freshly visible
    it fails with "No network with SSID found" or builds a profile with no
    ``key-mgmt``. We rescan first, and delete any incomplete profile a previous
    failed attempt left under the same name (a common cause of the key-mgmt
    error on retry), then connect fresh.
    """
    # Active rescan so the AP is visible to NetworkManager. Timeouts are kept
    # tight so the whole flow stays under the client's 35s notification timeout.
    try:
        subprocess.run(
            ["nmcli", "device", "wifi", "rescan"],
            capture_output=True, text=True, timeout=10,
        )
        time.sleep(1.5)  # let the scan results populate
    except Exception:
        pass  # best-effort

    # Drop any stale/half-built profile of the same name from a prior attempt.
    subprocess.run(
        ["nmcli", "connection", "delete", ssid],
        capture_output=True, text=True,
    )

    # Create the profile EXPLICITLY as WPA-PSK rather than letting
    # `nmcli device wifi connect` infer the security from the scan: that
    # inference fails (profile built with no key-mgmt → "key-mgmt is missing")
    # whenever the AP isn't freshly in the scan cache. Setting key-mgmt by hand
    # is deterministic for the common WPA/WPA2-PSK case.
    #
    # psk-flags 0 (NM_SETTING_SECRET_FLAG_NONE) forces the PSK to be stored in
    # the system connection. Without it the secret can end up "agent-owned",
    # and activation then fails headless with "Secrets were required, but not
    # provided" because there is no secret agent to ask.
    try:
        add = subprocess.run(
            [
                "nmcli", "connection", "add", "type", "wifi",
                "con-name", ssid, "ssid", ssid,
                "wifi-sec.key-mgmt", "wpa-psk",
                "wifi-sec.psk", password,
                "wifi-sec.psk-flags", "0",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if add.returncode != 0:
            return f"ERROR: {add.stderr.strip() or add.stdout.strip()}"

        up = subprocess.run(
            ["nmcli", "connection", "up", ssid],
            capture_output=True, text=True, timeout=20,
        )
        if up.returncode == 0:
            return f"OK: Connecting to {ssid}"
        # Bring-up failed (wrong password, out of range…): remove the profile
        # so the next attempt starts clean.
        error = up.stderr.strip() or up.stdout.strip()
        subprocess.run(
            ["nmcli", "connection", "delete", ssid],
            capture_output=True, text=True,
        )
        return f"ERROR: {error}"
    except subprocess.TimeoutExpired:
        return "ERROR: Connection timed out"
    except Exception as e:
        return f"ERROR: {e}"


def _wifi_reset() -> str:
    """Delete all saved WiFi connections (except Hotspot) via nmcli."""
    try:
        # List all 802-11-wireless connections
        result = subprocess.run(
            ["nmcli", "--escape", "yes", "-t", "-f", "NAME,TYPE", "connection", "show"],
            capture_output=True, text=True,
        )
        deleted = 0
        for line in result.stdout.splitlines():
            if ":802-11-wireless" not in line:
                continue
            conn_name = line.split(":802-11-wireless")[0]
            # Unescape nmcli escaping
            conn_name = conn_name.replace("\\:", ":")
            if conn_name == "Hotspot":
                continue
            subprocess.run(
                ["nmcli", "connection", "delete", conn_name],
                capture_output=True, text=True,
            )
            deleted += 1
        return f"OK: WiFi connections cleared ({deleted} removed)"
    except Exception as e:
        return f"ERROR: {e}"


# =====================================================================
# Main service class
# =====================================================================

class BluetoothWifiService:
    """BLE GATT service for WiFi configuration.

    Commands (written to COMMAND characteristic as UTF-8):
        PING                  → PONG
        PIN_xxxxx             → OK: Connected / ERROR: Incorrect PIN
        WIFI_SCAN             → JSON array of nearby SSIDs / ERROR: ...
        WIFI_KEYEX            → {"kid","pk","alg"} ephemeral pubkey for sealing
        WIFI_CONNECT_ENC json → OK: Connecting to <ssid> / ERROR: ...
        WIFI_RESET            → OK: WiFi connections cleared / ERROR: ...

    The WiFi password is sealed client-side (see _wifi_connect_enc) and never
    sent in clear — there is no plaintext connect command.

    PIN authentication is required before WIFI_SCAN/WIFI_CONNECT_ENC/WIFI_RESET.
    Auth is consumed by WIFI_CONNECT_ENC/WIFI_RESET (re-PIN for each) but NOT by
    WIFI_SCAN, so a client can scan then connect with a single PIN. WIFI_KEYEX
    is public (it returns only a public key). Auth is reset when the BLE central
    disconnects. Network status is readable from the STATUS service (every 10s).
    """

    def __init__(self, device_name: str = "Gripette", pin_code: str = "00000"):
        self.device_name = device_name
        # Name shown in the browser's device chooser. Includes the hostname so
        # several robots of the same type are distinguishable; the web client
        # filters by the "{device_name}" prefix, so this MUST start with it.
        self.advertised_name = f"{device_name} ({socket.gethostname()})"
        self.pin_code = pin_code
        self.authenticated = False
        # PIN brute-force state (see MAX_PIN_ATTEMPTS). Guarded by _pin_lock
        # because each BLE write is dispatched on its own daemon thread, so
        # concurrent PIN_ writes would otherwise race the counter. Held here on
        # the service (not per-connection) so it survives reconnects.
        self._pin_lock = threading.Lock()
        self._pin_failures = 0
        self._pin_lockout_rounds = 0
        self._pin_lockout_until = 0.0
        self.bus = None
        self.app = None
        self.mainloop = None
        # Adapter (hciN) index used for the MGMT advertising commands, and the
        # object path of the currently-connected central. The latter is used to
        # reset auth and re-assert advertising when a central drops (including
        # ungraceful drops like an app crash) so the device stays reconnectable.
        self._hci_index = None
        self._adapter_iface = None
        self._connected_device_path = None
        # Serializes advertise attempts so a disconnect-triggered re-advertise
        # can't overlap an in-progress (retrying) attempt.
        self._adv_lock = threading.Lock()
        # Ephemeral X25519 key for sealing the WiFi password. Regenerated on
        # each WIFI_KEYEX; only one central connects at a time and KEYEX is
        # immediately followed by CONNECT_ENC, so a single current key is
        # enough — and gives fresh per-exchange forward secrecy. kid lets a
        # stale CONNECT_ENC be rejected cleanly.
        self._ephemeral_key = None
        self._ephemeral_kid = 0

    def _handle_command(self, value: bytes) -> str:
        """Dispatch a BLE command and return response string."""
        command_str = value.decode("utf-8").strip()
        logger.info("Received command: %s", command_str)

        upper = command_str.upper()

        # PING — always allowed
        if upper == "PING":
            return "PONG"

        # PIN_xxxxx — authenticate (rate-limited; see _check_pin)
        if upper.startswith("PIN_"):
            return self._check_pin(command_str[4:].strip())

        # WIFI_SCAN — list nearby networks (requires auth; does NOT consume it,
        # so the client can scan then connect with a single PIN)
        if upper == "WIFI_SCAN":
            if not self.authenticated:
                return "ERROR: Not authenticated. Send PIN_xxxxx first."
            return _wifi_scan()

        # WIFI_KEYEX — hand out the ephemeral public key for sealing. Public:
        # a public key leaks nothing, and the client needs it before it can PIN.
        if upper == "WIFI_KEYEX":
            return self._wifi_keyex()

        # WIFI_CONNECT_ENC <json> — sealed connect (requires auth)
        if upper.startswith("WIFI_CONNECT_ENC"):
            if not self.authenticated:
                return "ERROR: Not authenticated. Send PIN_xxxxx first."
            parts = command_str.split(" ", 1)
            if len(parts) < 2:
                return "ERROR: Usage: WIFI_CONNECT_ENC <json>"
            self.authenticated = False  # one-shot auth
            return self._wifi_connect_enc(parts[1])

        # WIFI_RESET — requires auth
        if upper == "WIFI_RESET":
            if not self.authenticated:
                return "ERROR: Not authenticated. Send PIN_xxxxx first."
            self.authenticated = False  # one-shot auth
            return _wifi_reset()

        return f"ERROR: Unknown command: {command_str}"

    def _check_pin(self, pin: str) -> str:
        """Validate the PIN with brute-force rate limiting.

        While locked out, every attempt is refused WITHOUT checking the PIN (so
        the lockout can't be sidestepped, and a guess made during the window is
        never tested). A correct PIN clears all counters; each block of
        MAX_PIN_ATTEMPTS wrong guesses arms the next, longer lockout. See
        MAX_PIN_ATTEMPTS for the persistence/escalation rationale.
        """
        with self._pin_lock:
            now = time.monotonic()
            remaining = self._pin_lockout_until - now
            if remaining > 0:
                logger.warning(
                    "PIN attempt rejected — locked out for %ds more", int(remaining)
                )
                return f"ERROR: Too many attempts. Locked for {int(remaining) + 1}s."

            if pin == self.pin_code:
                self._pin_failures = 0
                self._pin_lockout_rounds = 0
                self.authenticated = True
                return "OK: Connected"

            self._pin_failures += 1
            if self._pin_failures < MAX_PIN_ATTEMPTS:
                return "ERROR: Incorrect PIN"

            # Threshold reached — arm the next lockout window (doubling each time)
            # and reset the per-window counter so the next block must re-earn it.
            self._pin_lockout_rounds += 1
            lockout = min(
                PIN_LOCKOUT_SECONDS * (2 ** (self._pin_lockout_rounds - 1)),
                MAX_PIN_LOCKOUT_SECONDS,
            )
            self._pin_lockout_until = now + lockout
            self._pin_failures = 0
            logger.warning(
                "PIN brute-force lockout #%d: %ds after %d wrong attempts",
                self._pin_lockout_rounds, lockout, MAX_PIN_ATTEMPTS,
            )
            return f"ERROR: Too many attempts. Locked for {lockout}s."

    # ---- WiFi password sealing ----

    def _wifi_keyex(self) -> str:
        """Generate a fresh ephemeral key, return it as JSON: {kid, pk(b64), alg}."""
        self._ephemeral_kid += 1
        self._ephemeral_key = X25519PrivateKey.generate()
        pk = self._ephemeral_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )
        return json.dumps(
            {
                "kid": str(self._ephemeral_kid),
                "pk": base64.b64encode(pk).decode(),
                "alg": KEYEX_ALG,
            },
            separators=(",", ":"),
        )

    def _wifi_connect_enc(self, blob: str) -> str:
        """Decrypt a sealed WIFI_CONNECT_ENC payload, then connect.

        Payload JSON: {ssid, kid, epk(b64 32B), nonce(b64 12B), ct(b64 ct||tag)}.
        Key = HKDF-SHA256(ecdh, salt=PIN, info=HKDF_INFO, 32B); AES-256-GCM with
        AAD=ssid. A wrong PIN, tampered ciphertext or stale key all surface as a
        single opaque decrypt failure (no oracle on which part was wrong).
        """
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            return "ERROR: Invalid payload (expected JSON)"
        try:
            ssid = data["ssid"]
            kid = data["kid"]
            epk = base64.b64decode(data["epk"])
            nonce = base64.b64decode(data["nonce"])
            ct = base64.b64decode(data["ct"])
        except (KeyError, TypeError, ValueError):
            return "ERROR: Malformed encrypted payload"

        if self._ephemeral_key is None or kid != str(self._ephemeral_kid):
            return "ERROR: Stale key — re-run WIFI_KEYEX"
        priv = self._ephemeral_key
        try:
            shared = priv.exchange(X25519PublicKey.from_public_bytes(epk))
            key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=self.pin_code.encode("utf-8"),
                info=HKDF_INFO,
            ).derive(shared)
            password = AESGCM(key).decrypt(
                nonce, ct, ssid.encode("utf-8")
            ).decode("utf-8")
        except Exception:
            return "ERROR: Decryption failed (wrong PIN?)"
        return _wifi_connect(ssid, password)

    def start(self):
        """Initialize BlueZ DBus objects and start advertising."""
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.bus = dbus.SystemBus()

        # Register pairing agent
        agent_manager = dbus.Interface(
            self.bus.get_object("org.bluez", "/org/bluez"),
            "org.bluez.AgentManager1",
        )
        self.agent = NoInputAgent(self.bus, AGENT_PATH)
        agent_manager.RegisterAgent(AGENT_PATH, "NoInputNoOutput")
        agent_manager.RequestDefaultAgent(AGENT_PATH)
        logger.info("BLE agent registered (Just Works pairing)")

        # Find and configure adapter
        adapter = self._find_adapter()
        if not adapter:
            raise RuntimeError("No Bluetooth adapter found")

        adapter_props = dbus.Interface(adapter, DBUS_PROP_IFACE)
        # Kept for RemoveDevice() on disconnect (see _remove_bond).
        self._adapter_iface = dbus.Interface(adapter, "org.bluez.Adapter1")
        adapter_props.Set("org.bluez.Adapter1", "Powered", dbus.Boolean(True))
        # Adapter alias = advertised name (e.g. "Grabette (grabette-01)") so the
        # device chooser shows the hostname and several robots are distinguishable.
        adapter_props.Set("org.bluez.Adapter1", "Alias", dbus.String(self.advertised_name))
        adapter_props.Set("org.bluez.Adapter1", "Discoverable", dbus.Boolean(True))
        adapter_props.Set(
            "org.bluez.Adapter1", "DiscoverableTimeout", dbus.UInt32(0)
        )
        # Pairable=True is REQUIRED for cross-platform reliability. The GATT
        # characteristics are unencrypted, so a central CAN use them
        # "connection-only" without bonding (this is what macOS does — which is
        # why Pairable=False appeared to work). But other stacks — notably
        # Windows, and some Linux/BlueZ centrals — INSIST on bonding before any
        # GATT operation. With Pairable=False BlueZ refuses their SMP pairing
        # request and the central drops the link ~0.5s after connect, looping
        # forever (observed as "Device disconnected / GATT Server is
        # disconnected" with connect→disconnect cycles every ~0.5s).
        #
        # With Pairable=True and the NoInputNoOutput agent above, those centrals
        # complete a SILENT Just Works bond (the SMP IO-capability mapping always
        # degrades to Just Works when either side is NoInputNoOutput — numeric
        # comparison can never be selected, so no confirmation dialog is raised),
        # while connection-only centrals are unaffected.
        adapter_props.Set("org.bluez.Adapter1", "Pairable", dbus.Boolean(True))

        # Register GATT application
        service_manager = dbus.Interface(adapter, GATT_MANAGER_IFACE)
        self.app = Application(self.bus, self._handle_command)
        service_manager.RegisterApplication(
            self.app.get_path(),
            {},
            reply_handler=lambda: logger.info("GATT application registered"),
            error_handler=lambda e: logger.error("Failed to register GATT app: %s", e),
        )

        # Start advertising via the legacy MGMT path (see _mgmt_advertise for
        # why we bypass bluetoothd's RegisterAdvertisement). No service UUID is
        # advertised: the web client filters by name prefix and reaches the
        # services via optionalServices after connecting.
        self._start_advertising()

        # Watch central connect/disconnect. BlueZ emits PropertiesChanged on
        # org.bluez.Device1 with Connected=true/false; we use the false edge to
        # reset the session and re-assert advertising — crucially including
        # UNGRACEFUL drops (app crash), where advertising would otherwise not
        # resume and the device becomes unreconnectable until a restart.
        self.bus.add_signal_receiver(
            self._on_device_properties_changed,
            dbus_interface=DBUS_PROP_IFACE,
            signal_name="PropertiesChanged",
            arg0="org.bluez.Device1",
            path_keyword="path",
        )

        # Periodic network status refresh (every 10s)
        GLib.timeout_add_seconds(10, self.app.status_service.update_network_status)

        logger.info("Bluetooth service started as '%s'", self.advertised_name)

    def _on_device_properties_changed(
        self, interface, changed, invalidated, path=None
    ):
        """React to BlueZ Device1 connect/disconnect transitions."""
        if interface != "org.bluez.Device1" or "Connected" not in changed:
            return
        if bool(changed["Connected"]):
            self._connected_device_path = path
            logger.info("BLE central connected: %s", path)
        else:
            logger.info("BLE central disconnected: %s", path)
            # Only act on the device we tracked, so a stale disconnect signal
            # can't clobber a client that just reconnected.
            if self._connected_device_path in (None, path):
                self._connected_device_path = None
                self._on_central_disconnected(path)

    def _on_central_disconnected(self, device_path):
        """Reset auth, drop the bond, and re-assert advertising after a drop."""
        self.authenticated = False
        # Forget the central's bond so the next session pairs fresh (see
        # _remove_bond), then re-advertise: a connectable advertisement stops
        # once a central connects, so it must be re-added to stay reconnectable
        # — including after ungraceful drops.
        self._remove_bond(device_path)
        self._start_advertising()

    def _remove_bond(self, device_path):
        """Remove the disconnected central's pairing/bond via Adapter1.RemoveDevice.

        Web Bluetooth clients are unreliable about persisting LE bonds: when the
        client forgets its key but we keep ours, the next connection re-pairs
        from scratch and the stale bond on our side makes SC pairing fail
        ("numeric comparison failed"), dropping the link in a connect→disconnect
        loop. We don't rely on the bond for security — the GATT characteristics
        are unencrypted and WiFi-password secrecy comes from the in-app
        PIN+X25519 sealing, not the BLE link — so clearing it after every
        session is safe and keeps reconnection robust across reboots and
        key-forgetting clients. Centrals that insist on bonding simply re-bond
        (silent Just Works) next time.
        """
        if self._adapter_iface is None or device_path is None:
            return
        try:
            self._adapter_iface.RemoveDevice(device_path)
            logger.info("Removed bond for %s", device_path)
        except dbus.exceptions.DBusException as e:
            logger.warning("RemoveDevice failed (non-fatal): %s", e)

    def _start_advertising(self):
        """Kick off (re-)advertising on a daemon thread.

        Never run btmgmt on the calling thread: at startup that thread is the
        one about to enter the GLib mainloop, and btmgmt can block for a long
        time (see _mgmt_advertise), which would stop the mainloop from ever
        running — leaving GATT unregistered and DBus unserviced.
        """
        threading.Thread(target=self._mgmt_advertise, daemon=True).start()

    def _mgmt_advertise(self):
        """(Re-)register a connectable LE advertisement via the legacy MGMT path.

        Works around a kernel-6.18 regression: bluetoothd's
        LEAdvertisingManager1.RegisterAdvertisement drives the controller through
        the *extended* advertising MGMT commands (Add Ext Adv Params/Data), which
        this controller (BCM4345C0 — no HCI extended advertising) rejects with
        "Invalid Parameters". The *legacy* MGMT "Add Advertising" command still
        works, so we drive it directly via btmgmt. The GATT server stays
        registered through bluetoothd (GattManager1.RegisterApplication is
        unaffected); a central connecting to this connectable advert reaches it
        normally.

        Runs on a daemon thread (see _start_advertising); the lock keeps a
        disconnect-triggered re-advertise from overlapping the initial one.
        """
        if not self._adv_lock.acquire(blocking=False):
            return  # an advertise attempt is already running
        try:
            # Clear any prior/lingering instance, then add ours: -c connectable,
            # -g general-discoverable, with the name in the scan response (-s)
            # because the legacy MGMT path doesn't read the adapter alias.
            self._btmgmt("rm-adv", "1")
            out = self._btmgmt(
                "add-adv", "-c", "-g", "-s", self._advertised_scan_rsp_hex(), "1"
            )
            if out and "Instance added" in out:
                logger.info("BLE advertisement registered via MGMT (legacy path)")
            else:
                logger.error("Failed to register advertisement via MGMT")
        finally:
            self._adv_lock.release()

    def _btmgmt(self, *args):
        """Run a btmgmt subcommand; return stdout (or None on timeout).

        btmgmt is built on bt_shell, which HANGS when its stdin is /dev/null —
        exactly what systemd gives a service. We hand it the read end of a pipe
        whose write end we keep open: stdin never reaches EOF, so bt_shell runs
        the one-shot command and exits instead of stalling. The timeout is a
        belt-and-braces guard so a wedged btmgmt can never block the caller.
        """
        r_fd, w_fd = os.pipe()
        try:
            result = subprocess.run(
                ["btmgmt", "--index", str(self._hci_index), *args],
                capture_output=True, text=True, timeout=5, stdin=r_fd,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            logger.warning("btmgmt %s timed out", " ".join(args))
            return None
        finally:
            os.close(r_fd)
            os.close(w_fd)

    def _advertised_scan_rsp_hex(self):
        """Scan-response payload as hex: a single Complete Local Name (0x09) AD.

        Truncated so the whole AD structure fits a 31-byte scan response
        (1 length byte + 1 type byte + <=29 name bytes). The web client filters
        by name prefix, so the leading "{device_name}" is preserved.
        """
        name = self.advertised_name.encode("utf-8")[:29]
        return f"{len(name) + 1:02x}09{name.hex()}"

    def _find_adapter(self):
        """Find the first BlueZ adapter that supports GATT + LE advertising."""
        remote_om = dbus.Interface(
            self.bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE
        )
        objects = remote_om.GetManagedObjects()
        for path, props in objects.items():
            if GATT_MANAGER_IFACE in props and LE_ADVERTISING_MANAGER_IFACE in props:
                # Remember the controller index (…/hciN) for the MGMT advertising
                # commands issued via btmgmt in _mgmt_advertise.
                self._hci_index = int(path.rsplit("/hci", 1)[1])
                return self.bus.get_object(BLUEZ_SERVICE_NAME, path)
        return None

    def run(self):
        """Start the service and block on the GLib main loop."""
        self.start()
        self.mainloop = GLib.MainLoop()
        try:
            logger.info("Running (Ctrl+C to exit)...")
            self.mainloop.run()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.mainloop.quit()
