"""Power off and Reboot only answer "on it" when sudo will run the command
without a password. `sudo -n -l <cmd>` is not that check: a user in the sudo
group passes it, then `sudo -n <cmd>` stops on the password — the dashboard
said "Rebooting…" and nothing happened. A fake `sudo` on PATH plays such a
device: it lists the grants it is given and logs what it runs."""
import asyncio
import os
import stat

import pytest
from fastapi import HTTPException

from grabette.app.routers import system

REBOOT = ("/usr/bin/systemctl", "reboot")

# The device of the bug report: in the sudo group, NOPASSWD only for poweroff.
_SUDO_GROUP_LISTING = """\
Matching Defaults entries for rasp on grabette:
    env_reset, mail_badpass,
    secure_path=/usr/local/sbin\\:/usr/local/bin\\:/usr/sbin\\:/usr/bin\\:/sbin\\:/bin,
    use_pty

User rasp may run the following commands on grabette:
    (ALL : ALL) ALL
    (root) NOPASSWD: /usr/bin/systemctl poweroff
"""


@pytest.mark.parametrize("line, granted", [
    ("(root) NOPASSWD: /usr/bin/systemctl reboot", True),
    ("(ALL) NOPASSWD: ALL", True),                       # Raspberry Pi OS default
    ("(ALL : ALL) NOPASSWD: ALL", True),
    ("(root) NOPASSWD: /usr/bin/systemctl", True),       # any arguments
    ("(root) NOPASSWD: /usr/sbin/modprobe i2c-dev, /usr/bin/systemctl reboot", True),
    ("(root) NOPASSWD: SETENV: /usr/bin/systemctl reboot", True),
    ("(ALL : ALL) ALL", False),                          # allowed, with a password
    ("(root) /usr/bin/systemctl reboot", False),
    ("(root) NOPASSWD: /usr/bin/systemctl poweroff", False),
    ("(root) NOPASSWD: /usr/bin/true, PASSWD: /usr/bin/systemctl reboot", False),
    ("(www-data) NOPASSWD: ALL", False),                 # not as root
])
def test_listing_parse(line, granted):
    assert system.sudo_grants_without_password(
        f"User rasp may run the following commands on g:\n    {line}\n",
        REBOOT) is granted


def test_sudo_group_alone_is_not_a_grant():
    assert not system.sudo_grants_without_password(_SUDO_GROUP_LISTING, REBOOT)
    assert system.sudo_grants_without_password(_SUDO_GROUP_LISTING,
                                               system.POWEROFF_CMD)


def _fake_sudo(tmp_path, monkeypatch, listing: str | None):
    """listing None: `sudo -l` wants a password too (no NOPASSWD grant)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "listing").write_text(listing or "")
    log = tmp_path / "log"
    sudo = bin_dir / "sudo"
    sudo.write_text(f"""#!/bin/bash
echo "$*" >> {log}
if [ "$*" = "-n -l" ]; then
  {"cat " + str(tmp_path / "listing") + "; exit 0" if listing is not None
   else 'echo "sudo: a password is required" >&2; exit 1'}
fi
exit 0
""")
    sudo.chmod(sudo.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    # No daemon: nothing recording to protect.
    monkeypatch.setattr("grabette.app.main.get_daemon_instance", lambda: None)
    return log


def _reboot_then_wait():
    async def run():
        try:
            return await system.system_reboot()
        finally:
            await asyncio.gather(*system._pending_sudo)
    return asyncio.run(run())


def test_reboot_refused_when_only_the_sudo_group_allows_it(tmp_path, monkeypatch):
    log = _fake_sudo(tmp_path, monkeypatch, _SUDO_GROUP_LISTING)
    with pytest.raises(HTTPException) as e:
        _reboot_then_wait()
    assert e.value.status_code == 500
    assert "make install-reboot" in e.value.detail
    assert log.read_text().splitlines() == ["-n -l"]  # never tried to reboot


def test_reboot_refused_without_any_grant(tmp_path, monkeypatch):
    _fake_sudo(tmp_path, monkeypatch, None)
    with pytest.raises(HTTPException) as e:
        _reboot_then_wait()
    assert "make install-reboot" in e.value.detail


def test_reboot_runs_the_exact_granted_command(tmp_path, monkeypatch):
    log = _fake_sudo(tmp_path, monkeypatch, _SUDO_GROUP_LISTING
                     + "    (root) NOPASSWD: /usr/bin/systemctl reboot\n")
    real = system.sudo_later
    monkeypatch.setattr(system, "sudo_later",
                        lambda cmd, delay_s=1.0: real(cmd, 0))  # no 1 s wait
    assert _reboot_then_wait() == {"status": "rebooting"}
    assert log.read_text().splitlines() == ["-n -l",
                                            "-n /usr/bin/systemctl reboot"]
