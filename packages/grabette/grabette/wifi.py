"""WiFi management helpers using nmcli.

Used by:
- app/routers/wifi.py to serve status and connect to networks from the web UI
"""

from __future__ import annotations

import logging
import subprocess
import time

logger = logging.getLogger(__name__)

HOTSPOT_CONN_NAME = "grabette-hotspot"
HOTSPOT_IFACE = "wlan0"
WIFI_CONN_NAME = "grabette-wifi"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def get_active_wifi_connection() -> str | None:
    """Return the NetworkManager connection name active on wlan0, or None."""
    result = _run(["nmcli", "-t", "-f", "device,connection", "dev", "status"])
    for line in result.stdout.splitlines():
        if line.startswith(f"{HOTSPOT_IFACE}:"):
            conn = line[len(HOTSPOT_IFACE) + 1:]
            return conn if conn else None
    return None


def get_network_mode() -> str:
    """Return 'hotspot', 'connected', or 'offline'."""
    conn = get_active_wifi_connection()
    if conn is None:
        return "offline"
    if conn == HOTSPOT_CONN_NAME:
        return "hotspot"
    return "connected"


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


def get_local_ip() -> str | None:
    """Return the WiFi interface's current IPv4 address, or None."""
    result = _run(["nmcli", "-g", "IP4.ADDRESS", "device", "show", HOTSPOT_IFACE])
    for line in result.stdout.strip().splitlines():
        if "/" in line:
            return line.split("/")[0]
    return None


# ---------------------------------------------------------------------------
# Network scanning and connection
# ---------------------------------------------------------------------------

def scan_networks() -> list[dict]:
    """Return visible WiFi networks sorted by signal, excluding the current connection."""
    own_ssid = get_current_ssid() or ""
    # Trigger the scan separately: --rescan yes on 'list' causes NM to return an
    # empty list when it throttles consecutive forced scans. We call 'rescan'
    # first (blocks until NM finishes), then read the updated cache with
    # --rescan no. If rescan fails (permission, interface busy), fall back to
    # --rescan auto so at least cached data is shown.
    rescan = _run(["nmcli", "dev", "wifi", "rescan"], timeout=10)
    if rescan.returncode != 0:
        logger.warning("wifi rescan failed (rc=%d): %s", rescan.returncode, rescan.stderr.strip())
    rescan_flag = "no" if rescan.returncode == 0 else "auto"

    # nmcli dev wifi rescan may return before the radio scan finishes (driver-
    # dependent). Retry listing up to 3 times with a short wait so we don't
    # return an empty list just because the cache hasn't been populated yet.
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
        logger.debug("wifi list attempt %d returned empty, retrying", attempt + 1)
    return []


def _delete_connections_for_ssid(ssid: str) -> None:
    """Remove any saved NM connection profiles for the given SSID.

    Stale profiles can have an incomplete 802-11-wireless-security section
    (key-mgmt missing) which causes nmcli device wifi connect to fail even
    when the credentials are correct.
    """
    result = _run(["nmcli", "--escape", "no", "-t", "-g",
                   "name,802-11-wireless.ssid", "connection", "show"])
    for line in result.stdout.splitlines():
        name, _, conn_ssid = line.partition(":")
        if conn_ssid.strip() == ssid:
            logger.info("[wifi] deleting stale profile %r for ssid %r", name, ssid)
            _run(["nmcli", "connection", "delete", name])


def wifi_connect(ssid: str, password: str) -> str:
    """Connect to a WiFi network. Returns a status string starting with 'OK:' or 'ERROR:'."""
    _delete_connections_for_ssid(ssid)
    _run(["nmcli", "connection", "delete", WIFI_CONN_NAME])

    # Build the profile explicitly so key-mgmt is never ambiguous.
    # nmcli device wifi connect relies on the NM scan cache to infer key-mgmt;
    # if the cache is stale or empty it creates an incomplete profile and fails.
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
    except Exception as exc:
        return f"ERROR: {exc}"
