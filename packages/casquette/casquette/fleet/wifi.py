# Trimmed from grabette/wifi.py (develop @0cb3453) for the casquette fleet
# prototype — only the two helpers relay_client needs (the device's routable IP
# and current SSID, reported to the fleet). Casquette does WiFi *provisioning*
# over BLE, not nmcli, so the scan/connect/hotspot machinery is intentionally
# omitted. Keep in sync until extracted to a shared package.
"""WiFi read helpers (nmcli) for fleet registration."""

from __future__ import annotations

import logging
import socket
import subprocess

logger = logging.getLogger(__name__)

# NetworkManager connection name for a device-hosted hotspot, if any. Used only
# to avoid reporting the hotspot itself as "the network this device is on".
HOTSPOT_CONN_NAME = "casquette-hotspot"
HOTSPOT_IFACE = "wlan0"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def get_active_wifi_connection() -> str | None:
    """Return the NetworkManager connection name active on wlan0, or None."""
    result = _run(["nmcli", "-t", "-f", "device,connection", "dev", "status"])
    for line in result.stdout.splitlines():
        if line.startswith(f"{HOTSPOT_IFACE}:"):
            conn = line[len(HOTSPOT_IFACE) + 1:]
            return conn if conn else None
    return None


def get_current_ssid() -> str | None:
    """Return the SSID of the current WiFi connection, or None."""
    # Primary: read from the active connection profile (reliable, no scan needed)
    conn = get_active_wifi_connection()
    if conn and conn != HOTSPOT_CONN_NAME:
        result = _run(["nmcli", "--escape", "no", "-g", "802-11-wireless.ssid",
                       "connection", "show", conn])
        ssid = result.stdout.strip()
        if ssid:
            return ssid

    # Fallback: scan-based approach
    result = _run(["nmcli", "--escape", "no", "-t", "-f", "active,ssid", "dev", "wifi"])
    for line in result.stdout.splitlines():
        if line.startswith("yes:"):
            return line[4:] or None
    return None


def get_route_ip() -> str:
    """Best-effort routable LAN IPv4 (the address an operator would use to reach
    this device), or "" if the host has no route out.

    Opens a UDP socket toward a public address (no packets are actually sent)
    and reads back the local address the kernel would route through.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return ""
