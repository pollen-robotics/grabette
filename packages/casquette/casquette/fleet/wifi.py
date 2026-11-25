# Trimmed from grabette/wifi.py (develop @0cb3453) for casquette: the read
# helpers relay_client needs (routable IP, current SSID) plus the scan/connect
# pair the dashboard's Network page uses. First-time provisioning still happens
# over BLE; the hotspot machinery is intentionally omitted. Keep in sync until
# extracted to a shared package.
"""WiFi helpers (nmcli): fleet registration and the dashboard's Network page."""

from __future__ import annotations

import logging
import socket
import subprocess
import time

logger = logging.getLogger(__name__)

# NetworkManager connection name for a device-hosted hotspot, if any. Used only
# to avoid reporting the hotspot itself as "the network this device is on".
HOTSPOT_CONN_NAME = "casquette-hotspot"
HOTSPOT_IFACE = "wlan0"
# Profile name for a network chosen from the dashboard.
WIFI_CONN_NAME = "casquette-wifi"


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


def get_signal_dbm() -> int | None:
    """Signal of the current WiFi link in dBm (from /proc/net/wireless), or None."""
    try:
        with open("/proc/net/wireless") as f:
            for line in f:
                if line.strip().startswith(f"{HOTSPOT_IFACE}:"):
                    return int(float(line.split()[3]))
    except (OSError, ValueError, IndexError):
        pass
    return None


def scan_networks() -> list[dict]:
    """Visible WiFi networks sorted by signal, excluding the current one."""
    own_ssid = get_current_ssid() or ""
    # Rescan first, then read the cache with --rescan no: --rescan yes on
    # 'list' makes NM return an empty list when it throttles forced scans.
    rescan = _run(["nmcli", "dev", "wifi", "rescan"], timeout=10)
    if rescan.returncode != 0:
        logger.warning("wifi rescan failed (rc=%d): %s", rescan.returncode, rescan.stderr.strip())
    rescan_flag = "no" if rescan.returncode == 0 else "auto"

    # The rescan may return before the radio scan finishes: retry the listing.
    for attempt in range(3):
        if attempt > 0:
            time.sleep(2)
        try:
            result = _run(
                ["nmcli", "--escape", "no", "-t", "-f", "SSID,SIGNAL",
                 "dev", "wifi", "list", "--rescan", rescan_flag],
                timeout=10,
            )
        except subprocess.TimeoutExpired:
            logger.warning("wifi scan timed out")
            return []
        networks: list[dict] = []
        seen: set[str] = set()
        for line in result.stdout.splitlines():
            idx = line.rfind(":")
            if idx < 0:
                continue
            ssid = line[:idx].strip()
            if not ssid or ssid == own_ssid or ssid in seen:
                continue
            seen.add(ssid)
            try:
                signal = int(line[idx + 1:].strip())
            except ValueError:
                continue
            networks.append({"ssid": ssid, "signal": signal})
        if networks:
            return sorted(networks, key=lambda n: n["signal"], reverse=True)
    return []


def _delete_connections_for_ssid(ssid: str) -> None:
    """Remove saved NM profiles for this SSID: a stale one with an incomplete
    security section makes the connect fail even with the right password."""
    result = _run(["nmcli", "--escape", "no", "-t", "-g",
                   "name,802-11-wireless.ssid", "connection", "show"])
    for line in result.stdout.splitlines():
        name, _, conn_ssid = line.partition(":")
        if conn_ssid.strip() == ssid:
            logger.info("[wifi] deleting stale profile %r for ssid %r", name, ssid)
            _run(["nmcli", "connection", "delete", name])


def wifi_connect(ssid: str, password: str) -> str:
    """Connect to a WiFi network. Returns a string starting with 'OK:' or 'ERROR:'.

    On failure NetworkManager falls back to the previous saved network on its
    own, so the device stays reachable where it was.
    """
    _delete_connections_for_ssid(ssid)
    _run(["nmcli", "connection", "delete", WIFI_CONN_NAME])

    # Build the profile explicitly so key-mgmt is never inferred from a stale
    # scan cache.
    cmd = [
        "nmcli", "connection", "add", "type", "wifi",
        "con-name", WIFI_CONN_NAME,
        "ssid", ssid,
        "ifname", HOTSPOT_IFACE,
    ]
    if password:
        cmd += ["wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", password]

    try:
        result = _run(cmd, timeout=15)
        if result.returncode != 0:
            return f"ERROR: {result.stderr.strip() or result.stdout.strip()}"
        result = _run(
            ["nmcli", "connection", "up", WIFI_CONN_NAME, "ifname", HOTSPOT_IFACE],
            timeout=60,
        )
        if result.returncode == 0:
            return f"OK: Connected to {ssid}"
        error = result.stderr.strip() or result.stdout.strip()
        _run(["nmcli", "connection", "delete", WIFI_CONN_NAME])
        return f"ERROR: {error}"
    except subprocess.TimeoutExpired:
        return "ERROR: Connection timed out"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"
