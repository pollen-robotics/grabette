"""WiFi management helpers using nmcli.

Used by:
- app/routers/wifi.py to serve status and connect to networks from the web UI
- app/routers/system.py and relay_client.py for the device's routable LAN IP
- relay_client.py for the SSID reported to the fleet (which network this device is on)
"""

from __future__ import annotations

import logging
import socket
import subprocess
import time

logger = logging.getLogger(__name__)

HOTSPOT_CONN_NAME = "grabette-hotspot"
HOTSPOT_IFACE = "wlan0"
# NB: there is deliberately no single "grabette-wifi" profile name any more.
# Every dashboard-configured network used to be written into that one profile,
# so connecting to a second network erased the first and nothing was ever
# "known". Profiles are now named after the SSID — the same convention the
# Bluetooth provisioning service uses (bluetooth/bluetooth_service.py) — so both
# paths produce the same profiles and they accumulate. A leftover grabette-wifi
# profile from an older image still lists fine: list_saved_networks reads the
# SSID out of the profile rather than trusting its name.


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


def get_route_ip() -> str:
    """Best-effort routable LAN IPv4 (the address an operator would use to reach
    this device), or "" if the host has no route out.

    Opens a UDP socket toward a public address (no packets are actually sent)
    and reads back the local address the kernel would route through. Unlike
    ``get_local_ip`` this is interface-agnostic (works over Ethernet too).
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
    """Return visible WiFi networks sorted by signal, excluding known ones.

    Both the current connection and every already-saved profile are filtered
    out: those are offered as one-click switches by ``list_saved_networks``, and
    listing them here too would invite re-typing a password that is already
    stored.
    """
    own_ssid = get_current_ssid() or ""
    known = {n["ssid"] for n in list_saved_networks()}
    known.add(own_ssid)
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
            if not ssid or ssid in known or ssid in seen:
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


def list_saved_networks() -> list[dict]:
    """Return the WiFi networks this device already knows, current one first.

    A "known" network is any saved NetworkManager wifi profile — whichever path
    created it: the dashboard, the Bluetooth provisioning tool, or nmcli by
    hand. The hotspot is excluded: it is how you reach the device when it is on
    no network at all, not somewhere to switch to.

    Each entry is ``{"ssid", "name", "active"}`` — ``name`` is the NM profile
    name (what has to be passed to ``nmcli connection up``, and not always the
    SSID), ``active`` marks the one currently up.
    """
    result = _run(["nmcli", "--escape", "no", "-t", "-f", "NAME,TYPE",
                   "connection", "show"])
    active_conn = get_active_wifi_connection()
    by_ssid: dict[str, dict] = {}
    for line in result.stdout.splitlines():
        # Split from the RIGHT on the last ':': TYPE never contains one, a
        # profile NAME (often an SSID) very well may.
        name, _, conn_type = line.rpartition(":")
        if conn_type not in ("802-11-wireless", "wifi"):
            continue
        if name == HOTSPOT_CONN_NAME:
            continue
        # The SSID is read per-profile rather than as an extra -f column: mixing
        # property fields into a multi-row listing is not something nmcli
        # renders reliably, and the profile name is only usually the SSID.
        ssid = _run(["nmcli", "--escape", "no", "-g", "802-11-wireless.ssid",
                     "connection", "show", name]).stdout.strip() or name
        entry = {"ssid": ssid, "name": name, "active": name == active_conn}
        # Two profiles for one SSID is a leftover, not two networks to choose
        # between. Keep one — the active profile if either of them is it, so
        # the list never shows the network you are on as one you could switch to.
        if ssid not in by_ssid or entry["active"]:
            by_ssid[ssid] = entry
    return sorted(by_ssid.values(), key=lambda n: (not n["active"], n["ssid"].lower()))


def activate_saved_network(ssid: str) -> str:
    """Bring up an already-known network by SSID, without re-entering its password.

    Returns a status string starting with 'OK:' or 'ERROR:'.
    """
    match = next((n for n in list_saved_networks() if n["ssid"] == ssid), None)
    if match is None:
        return f"ERROR: {ssid} is not a known network on this device"
    try:
        result = _run(["nmcli", "connection", "up", match["name"]], timeout=60)
        if result.returncode == 0:
            return f"OK: Connected to {ssid}"
        # Deliberately NOT deleting the profile on failure, unlike a fresh
        # connect: the credentials are known-good, the network is simply out of
        # range or down, and forgetting it would defeat the whole point.
        return f"ERROR: {result.stderr.strip() or result.stdout.strip()}"
    except subprocess.TimeoutExpired:
        return "ERROR: Connection timed out"
    except Exception as exc:
        return f"ERROR: {exc}"


def wifi_connect(ssid: str, password: str) -> str:
    """Connect to a WiFi network. Returns a status string starting with 'OK:' or 'ERROR:'."""
    # Only profiles for THIS ssid are dropped — every other known network is
    # left alone so it can still be switched back to from the dashboard.
    _delete_connections_for_ssid(ssid)

    # Build the profile explicitly so key-mgmt is never ambiguous.
    # nmcli device wifi connect relies on the NM scan cache to infer key-mgmt;
    # if the cache is stale or empty it creates an incomplete profile and fails.
    cmd = [
        "nmcli", "connection", "add", "type", "wifi",
        "con-name", ssid,
        "ssid", ssid,
        "ifname", HOTSPOT_IFACE,
    ]
    if password:
        # psk-flags 0 (SECRET_FLAG_NONE) stores the PSK in the system
        # connection. Left agent-owned, a later re-activation of this profile
        # fails headless with "Secrets were required, but not provided" — which
        # is exactly what switching back to a known network does.
        cmd += ["wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", password,
                "wifi-sec.psk-flags", "0"]

    try:
        result = _run(cmd, timeout=15)
        if result.returncode != 0:
            return f"ERROR: {result.stderr.strip() or result.stdout.strip()}"

        result = _run(
            ["nmcli", "connection", "up", ssid, "ifname", HOTSPOT_IFACE],
            timeout=60,
        )
        if result.returncode == 0:
            return f"OK: Connected to {ssid}"
        error = result.stderr.strip() or result.stdout.strip()
        _run(["nmcli", "connection", "delete", ssid])
        return f"ERROR: {error}"
    except subprocess.TimeoutExpired:
        return "ERROR: Connection timed out"
    except Exception as exc:
        return f"ERROR: {exc}"
