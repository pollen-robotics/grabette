"""Known-network switching — the nmcli parsing and the profile-per-SSID rule.

The point of these is the invariant the dashboard's one-click switch rests on:
a network stays known after you leave it. Before profiles were named per SSID,
every connect overwrote the single ``grabette-wifi`` profile and there was
nothing to switch back to.
"""

from __future__ import annotations

import subprocess

from grabette import wifi


def _fake_run(responses):
    """Stub _run: match on the nmcli argv, return (stdout, rc) from `responses`.

    `responses` maps a substring of the joined command to (stdout, returncode).
    Calls are recorded on the returned function as `.calls`.
    """
    def run(cmd, **kwargs):
        joined = " ".join(cmd)
        run.calls.append(cmd)
        for needle, (out, rc) in responses.items():
            if needle in joined:
                return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    run.calls = []
    return run


CONN_LIST = (
    "Wired connection 1:802-3-ethernet\n"
    "grabette-hotspot:802-11-wireless\n"
    "Livebox-D2C6:802-11-wireless\n"
    "office wifi:802-11-wireless\n"
)


def test_lists_saved_wifi_and_hides_ethernet_and_hotspot(monkeypatch):
    ssids = {"Livebox-D2C6": "Livebox-D2C6", "office wifi": "OfficeNet"}

    def run(cmd, **kwargs):
        joined = " ".join(cmd)
        if "connection show" in joined and "-f NAME,TYPE" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=CONN_LIST, stderr="")
        if "802-11-wireless.ssid" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=ssids[cmd[-1]] + "\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(wifi, "_run", run)
    monkeypatch.setattr(wifi, "get_active_wifi_connection", lambda: "office wifi")

    nets = wifi.list_saved_networks()
    assert [n["ssid"] for n in nets] == ["OfficeNet", "Livebox-D2C6"]
    # The active one sorts first and is flagged, so the UI never offers a
    # "Switch" button for the network you are already on.
    assert nets[0]["active"] is True
    assert nets[1]["active"] is False
    # The profile name is kept separately: it is what nmcli needs, and it is
    # NOT always the SSID.
    assert nets[0]["name"] == "office wifi"


def test_scan_hides_networks_that_are_already_known(monkeypatch):
    monkeypatch.setattr(wifi, "get_current_ssid", lambda: "OfficeNet")
    monkeypatch.setattr(wifi, "list_saved_networks", lambda: [
        {"ssid": "OfficeNet", "name": "office wifi", "active": True},
        {"ssid": "Livebox-D2C6", "name": "Livebox-D2C6", "active": False},
    ])
    monkeypatch.setattr(wifi, "_run", _fake_run({
        "dev wifi rescan": ("", 0),
        "dev wifi list": ("OfficeNet:90\nLivebox-D2C6:70\nGuestNet:55\n", 0),
    }))
    assert wifi.scan_networks() == [{"ssid": "GuestNet", "signal": 55}]


def test_activate_brings_up_the_profile_by_its_nm_name(monkeypatch):
    monkeypatch.setattr(wifi, "list_saved_networks", lambda: [
        {"ssid": "OfficeNet", "name": "office wifi", "active": False},
    ])
    run = _fake_run({"connection up": ("", 0)})
    monkeypatch.setattr(wifi, "_run", run)

    assert wifi.activate_saved_network("OfficeNet").startswith("OK:")
    assert ["nmcli", "connection", "up", "office wifi"] in run.calls


def test_activate_keeps_the_profile_when_bring_up_fails(monkeypatch):
    """A known network that is merely out of range must stay known."""
    monkeypatch.setattr(wifi, "list_saved_networks", lambda: [
        {"ssid": "OfficeNet", "name": "OfficeNet", "active": False},
    ])
    run = _fake_run({"connection up": ("Error: no suitable device", 1)})
    monkeypatch.setattr(wifi, "_run", run)

    assert wifi.activate_saved_network("OfficeNet").startswith("ERROR:")
    assert not any("delete" in c for cmd in run.calls for c in cmd)


def test_activate_refuses_an_unknown_ssid(monkeypatch):
    monkeypatch.setattr(wifi, "list_saved_networks", lambda: [])
    monkeypatch.setattr(wifi, "_run", _fake_run({}))
    assert "not a known network" in wifi.activate_saved_network("Nope")


def test_connect_names_the_profile_after_the_ssid_and_spares_the_others(monkeypatch):
    """The regression that made "known networks" possible at all."""
    deleted: list[str] = []

    def fake_delete(ssid):
        deleted.append(ssid)

    run = _fake_run({"connection add": ("", 0), "connection up": ("", 0)})
    monkeypatch.setattr(wifi, "_delete_connections_for_ssid", fake_delete)
    monkeypatch.setattr(wifi, "_run", run)

    assert wifi.wifi_connect("OfficeNet", "hunter2hunter2").startswith("OK:")

    add = next(c for c in run.calls if "add" in c)
    assert add[add.index("con-name") + 1] == "OfficeNet"
    # The PSK must be system-owned or re-activating this profile later fails
    # headless with "Secrets were required, but not provided".
    assert add[add.index("wifi-sec.psk-flags") + 1] == "0"
    # Only this SSID's stale profiles go; the legacy single-profile delete that
    # wiped every other known network is gone.
    assert deleted == ["OfficeNet"]
    assert "grabette-wifi" not in [c for cmd in run.calls for c in cmd]


def test_failed_connect_removes_the_half_built_profile(monkeypatch):
    run = _fake_run({"connection add": ("", 0), "connection up": ("Secrets were required", 1)})
    monkeypatch.setattr(wifi, "_delete_connections_for_ssid", lambda s: None)
    monkeypatch.setattr(wifi, "_run", run)

    assert wifi.wifi_connect("OfficeNet", "wrongpass").startswith("ERROR:")
    assert ["nmcli", "connection", "delete", "OfficeNet"] in run.calls
