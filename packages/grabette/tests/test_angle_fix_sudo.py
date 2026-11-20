"""The Diagnose popup's fixes run as root through `sudo -n` and the root-owned
helper of `make install-i2c-fix` — never with a password. A fake `sudo` on
PATH stands in for the real one: it runs what it is granted (the helper's
actions, here a fake helper) and refuses the rest with the real sudo's
message."""
import asyncio
import os
import stat

import pytest
from fastapi import HTTPException

from grabette.app.routers import angle
from grabette.hardware import i2c_diag


def _exe(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def device(tmp_path, monkeypatch):
    """`granted`: the helper actions sudoers allows. The fake helper logs each
    action; "boom" fails like a real error of the action."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "log"
    granted = tmp_path / "granted"
    granted.write_text("")
    helper = tmp_path / "grabette-i2c-fix"
    _exe(helper, f"""#!/bin/bash
echo "$1" >> {log}
if [ "$1" = boom ]; then echo "modprobe: FATAL: no such module" >&2; exit 1; fi
if [ "$1" = enable-overlays ]; then echo /boot/firmware/config.txt; fi
""")
    _exe(bin_dir / "sudo", f"""#!/bin/bash
[ "$1" = "-n" ] || exit 99
shift
if [ "$1" = "{angle._HELPER}" ] && grep -qx "$2" {granted}; then
  exec {helper} "$2"
fi
echo "sudo: a password is required" >&2; exit 1
""")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    class Device:
        def grant(self, *actions):
            granted.write_text("".join(f"{a}\n" for a in actions))

        def ran(self):
            return log.read_text().split() if log.exists() else []
    return Device()


def test_not_granted_is_not_permitted(device):
    with pytest.raises(angle.NotPermitted):
        asyncio.run(angle._helper("load"))
    assert device.ran() == []


def test_granted_action_runs_and_returns_its_output(device):
    device.grant("enable-overlays")
    assert asyncio.run(angle._helper("enable-overlays")) == \
        "/boot/firmware/config.txt"
    assert device.ran() == ["enable-overlays"]


def test_a_failing_action_is_not_mistaken_for_a_permission_problem(device):
    device.grant("boom")
    with pytest.raises(RuntimeError, match="no such module"):
        asyncio.run(angle._helper("boom"))


class _Backend:
    angle_sensors_status = {"error": ""}

    def get_capture_status(self):
        class C:
            is_capturing = is_starting = False
        return C()

    def reinit_angle_sensors(self):
        pass


def test_fix_without_the_grant_says_what_to_install(device):
    with pytest.raises(HTTPException) as e:
        asyncio.run(angle.angle_fix(
            angle.FixRequest(fix=i2c_diag.FIX_PERSIST_I2C_DEV), _Backend()))
    assert e.value.status_code == 500
    assert "make install-i2c-fix" in e.value.detail


def test_fix_with_the_grant_runs_the_helper(device, monkeypatch):
    device.grant("persist")
    monkeypatch.setattr(angle, "_diagnose", lambda backend: {"healthy": True})
    res = asyncio.run(angle.angle_fix(
        angle.FixRequest(fix=i2c_diag.FIX_PERSIST_I2C_DEV), _Backend()))
    assert res["diagnosis"] == {"healthy": True}
    assert device.ran() == ["persist"]


def test_a_password_in_the_request_is_not_accepted():
    # The field is gone: an old dashboard sending one gets it ignored, never
    # passed to sudo.
    assert "password" not in angle.FixRequest.model_fields


# ── No fix button the device may not apply ────────────────────────────

def _listing(*commands):
    return "User rasp may run the following commands on g:\n    (ALL : ALL) ALL\n" + \
        "".join(f"    (root) NOPASSWD: {c}\n" for c in commands)


def _diag(fix, manual=("sudo reboot",)):
    return {"fix": fix, "fix_label": i2c_diag.FIX_LABELS.get(fix, ""),
            "summary": "Something is wrong.", "manual": list(manual)}


def test_ungranted_fix_is_replaced_by_its_make_target():
    diag = angle._without_ungranted_fix(_diag(i2c_diag.FIX_LOAD_I2C_DEV),
                                        _listing(f"{angle._HELPER} load"))
    assert diag["fix"] is None and diag["fix_label"] == ""
    assert "make install-i2c-fix" in diag["summary"]
    assert diag["manual"][0] == i2c_diag.make_cmd("install-i2c-fix")
    assert diag["manual"][-1] == "sudo reboot"  # the by-hand steps stay


def test_granted_fix_keeps_its_button():
    diag = _diag(i2c_diag.FIX_LOAD_I2C_DEV)
    listing = _listing(f"{angle._HELPER} load", f"{angle._HELPER} persist")
    assert angle._without_ungranted_fix(diag, listing) == diag


def test_reboot_fix_needs_the_reboot_grant():
    diag = angle._without_ungranted_fix(
        _diag(i2c_diag.FIX_REBOOT),
        _listing(f"{angle._HELPER} load", f"{angle._HELPER} persist",
                 f"{angle._HELPER} enable-overlays"))
    assert diag["fix"] is None
    assert diag["manual"][0] == i2c_diag.make_cmd("install-reboot")
    granted = _diag(i2c_diag.FIX_REBOOT)
    assert angle._without_ungranted_fix(
        granted, _listing("/usr/bin/systemctl reboot")) == granted


def test_no_grant_at_all_and_no_root_fix():
    assert angle._without_ungranted_fix(
        _diag(i2c_diag.FIX_PERSIST_I2C_DEV), "")["fix"] is None
    reinit = _diag(i2c_diag.FIX_REINIT)
    assert angle._without_ungranted_fix(reinit, "") == reinit
