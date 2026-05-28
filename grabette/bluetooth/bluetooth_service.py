"""Bluetooth LE WiFi configuration service for Grabette.

Exposes a BLE GATT service that allows configuring WiFi credentials
from a phone or laptop (via Web Bluetooth or any BLE client).

Adapted from reachy_mini bluetooth_service.py:
https://github.com/pollen-robotics/reachy_mini/tree/main/src/reachy_mini/daemon/app/services/bluetooth

Changes from reference:
- Device name: "Grabette"
- PIN: from env var GRABETTE_BT_PIN (not dfu-util serial)
- Direct WIFI/WIFI_RESET commands via nmcli (no CMD_ shell scripts)
- No Device Info Service (unnecessary)
- Simplified status service (network status only)
"""

import logging
import subprocess
import threading
from typing import Callable

import dbus
import dbus.mainloop.glib
import dbus.service
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
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"
AGENT_PATH = "/org/bluez/agent"

# Descriptor UUIDs
USER_DESCRIPTION_UUID = "00002901-0000-1000-8000-00805f9b34fb"


# =====================================================================
# BLE Agent — "Just Works" pairing (no user interaction on device side)
# =====================================================================

class NoInputAgent(dbus.service.Object):
    """BLE Agent for Just Works pairing (NoInputNoOutput capability)."""

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Release(self, *args):
        logger.info("Agent released")

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="s")
    def RequestPinCode(self, *args):
        logger.info("RequestPinCode — returning empty (Just Works)")
        return ""

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="u")
    def RequestPasskey(self, *args):
        logger.info("RequestPasskey — returning 0 (Just Works)")
        return dbus.UInt32(0)

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def RequestConfirmation(self, *args):
        logger.info("RequestConfirmation — auto-accepting")

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def DisplayPinCode(self, *args):
        pass

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def DisplayPasskey(self, *args):
        pass

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def AuthorizeService(self, *args):
        logger.info("AuthorizeService — auto-accepting")

    @dbus.service.method("org.bluez.Agent1", in_signature="", out_signature="")
    def Cancel(self, *args):
        logger.info("Agent request canceled")


# =====================================================================
# BLE Advertisement
# =====================================================================

class Advertisement(dbus.service.Object):
    """BLE peripheral advertisement."""

    PATH_BASE = "/org/bluez/advertisement"

    def __init__(self, bus, index, advertising_type, local_name):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.ad_type = advertising_type
        self.local_name = local_name
        self.service_uuids = None
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        props = {"Type": self.ad_type}
        if self.local_name:
            props["LocalName"] = dbus.String(self.local_name)
        if self.service_uuids:
            props["ServiceUUIDs"] = dbus.Array(self.service_uuids, signature="s")
        props["Appearance"] = dbus.UInt16(0x0000)
        props["Duration"] = dbus.UInt16(0)
        props["Timeout"] = dbus.UInt16(0)
        return {LE_ADVERTISEMENT_IFACE: props}

    def get_path(self):
        return dbus.ObjectPath(self.path)

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != LE_ADVERTISEMENT_IFACE:
            raise dbus.exceptions.DBusException(
                "org.freedesktop.DBus.Error.InvalidArgs",
                "Unknown interface " + interface,
            )
        return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE, in_signature="", out_signature="")
    def Release(self):
        logger.info("Advertisement released")


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
        # Update response char on the GLib main loop thread (thread-safe).
        def _update():
            encoded = [dbus.Byte(b) for b in response.encode("utf-8")]
            self.service.response_char.value = encoded
            # Notify subscribed BLE clients so they don't need to poll.
            self.service.response_char.PropertiesChanged(
                GATT_CHRC_IFACE,
                {"Value": dbus.Array(encoded, signature="y")},
                dbus.Array([], signature="s"),
            )
            logger.info("Command processed, response: %s", response)
            return False  # one-shot
        GLib.idle_add(_update)


class ResponseCharacteristic(Characteristic):
    """Read/notify characteristic that holds the last command response."""

    def __init__(self, bus, index, service):
        super().__init__(bus, index, RESPONSE_CHAR_UUID, ["read", "notify"], service)


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

def get_network_status() -> str:
    """Return network status string: "{MODE} [iface] ip ; [iface] ip".

    MODE is one of: CONNECTED, HOTSPOT, OFFLINE.
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
        return f"{mode} {' ; '.join(parts)}"
    except Exception as e:
        logger.error("Error getting network status: %s", e)
        return "ERROR"


def _wifi_connect(ssid: str, password: str) -> str:
    """Connect to a WiFi network using nmcli. Returns status message."""
    try:
        result = subprocess.run(
            ["nmcli", "device", "wifi", "connect", ssid, "password", password],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return f"OK: Connecting to {ssid}"
        else:
            error = result.stderr.strip() or result.stdout.strip()
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
        PING               → PONG
        PIN_xxxxx          → OK: Connected / ERROR: Incorrect PIN
        WIFI ssid password → OK: Connecting to <ssid> / ERROR: ...
        WIFI_RESET         → OK: WiFi connections cleared / ERROR: ...

    PIN authentication is required before WIFI/WIFI_RESET commands.
    Network status is readable from the STATUS service (updates every 10s).
    """

    def __init__(self, device_name: str = "Grabette", pin_code: str = "00000"):
        self.device_name = device_name
        self.pin_code = pin_code
        self.authenticated = False
        self.bus = None
        self.app = None
        self.adv = None
        self.mainloop = None

    def _handle_command(self, value: bytes) -> str:
        """Dispatch a BLE command and return response string."""
        command_str = value.decode("utf-8").strip()
        logger.info("Received command: %s", command_str)

        upper = command_str.upper()

        # PING — always allowed
        if upper == "PING":
            return "PONG"

        # PIN_xxxxx — authenticate
        if upper.startswith("PIN_"):
            pin = command_str[4:].strip()
            if pin == self.pin_code:
                self.authenticated = True
                return "OK: Connected"
            else:
                return "ERROR: Incorrect PIN"

        # WIFI ssid password — requires auth
        if upper.startswith("WIFI "):
            if not self.authenticated:
                return "ERROR: Not authenticated. Send PIN_xxxxx first."
            # Parse: "WIFI ssid password" (password may contain spaces)
            parts = command_str.split(" ", 2)
            if len(parts) < 3:
                return "ERROR: Usage: WIFI <ssid> <password>"
            ssid, password = parts[1], parts[2]
            self.authenticated = False  # one-shot auth
            return _wifi_connect(ssid, password)

        # WIFI_RESET — requires auth
        if upper == "WIFI_RESET":
            if not self.authenticated:
                return "ERROR: Not authenticated. Send PIN_xxxxx first."
            self.authenticated = False  # one-shot auth
            return _wifi_reset()

        return f"ERROR: Unknown command: {command_str}"

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
        adapter_props.Set("org.bluez.Adapter1", "Powered", dbus.Boolean(True))
        # Set the adapter alias so the device shows as "Gripette" (not hostname)
        adapter_props.Set("org.bluez.Adapter1", "Alias", dbus.String(self.device_name))
        adapter_props.Set("org.bluez.Adapter1", "Discoverable", dbus.Boolean(True))
        adapter_props.Set(
            "org.bluez.Adapter1", "DiscoverableTimeout", dbus.UInt32(0)
        )
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

        # Register BLE advertisement
        ad_manager = dbus.Interface(adapter, LE_ADVERTISING_MANAGER_IFACE)
        self.adv = Advertisement(self.bus, 0, "peripheral", self.device_name)
        self.adv.service_uuids = [STATUS_SERVICE_UUID]
        ad_manager.RegisterAdvertisement(
            self.adv.get_path(),
            {},
            reply_handler=lambda: logger.info("BLE advertisement registered"),
            error_handler=lambda e: logger.error(
                "Failed to register advertisement: %s", e
            ),
        )

        # Periodic network status refresh (every 10s)
        GLib.timeout_add_seconds(10, self.app.status_service.update_network_status)

        logger.info("Bluetooth service started as '%s'", self.device_name)

    def _find_adapter(self):
        """Find the first BlueZ adapter that supports GATT + LE advertising."""
        remote_om = dbus.Interface(
            self.bus.get_object(BLUEZ_SERVICE_NAME, "/"), DBUS_OM_IFACE
        )
        objects = remote_om.GetManagedObjects()
        for path, props in objects.items():
            if GATT_MANAGER_IFACE in props and LE_ADVERTISING_MANAGER_IFACE in props:
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
