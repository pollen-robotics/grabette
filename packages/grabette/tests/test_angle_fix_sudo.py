"""The Diagnose popup's fixes run as root through sudo, with the password typed
in the popup when sudo asks for one. A fake `sudo` on PATH stands in for the
real one: it wants "rasp", and answers with the real sudo's messages."""
import asyncio
import os
import stat

import pytest

from grabette.app.routers import angle

_FAKE_SUDO = r'''#!/bin/bash
args=("$@")
if [ "$1" = "-n" ]; then echo "sudo: a password is required" >&2; exit 1; fi
# -S -k -p "" <cmd...>
IFS= read -r pw
if [ "$pw" != "rasp" ]; then
  printf 'Sorry, try again.\nsudo: 1 incorrect password attempt\n' >&2; exit 1
fi
exec "${args[@]:4}"
'''


@pytest.fixture
def fake_sudo(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sudo = bin_dir / "sudo"
    sudo.write_text(_FAKE_SUDO)
    sudo.chmod(sudo.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return tmp_path


def test_no_password_asks_for_one(fake_sudo):
    with pytest.raises(angle.PasswordRequired) as e:
        asyncio.run(angle._sudo(("/usr/bin/true",)))
    assert not e.value.wrong


def test_wrong_password_says_so(fake_sudo):
    with pytest.raises(angle.PasswordRequired) as e:
        asyncio.run(angle._sudo(("/usr/bin/true",), password="nope"))
    assert e.value.wrong


def test_right_password_runs_the_command_with_the_rest_of_stdin(fake_sudo):
    out = fake_sudo / "i2c-dev.conf"
    asyncio.run(angle._sudo(("/usr/bin/tee", str(out)), "i2c-dev\n",
                            password="rasp"))
    # The password line went to sudo, only the payload to the command.
    assert out.read_text() == "i2c-dev\n"


def test_a_failing_command_is_not_mistaken_for_a_password_problem(fake_sudo):
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(angle._sudo(("/usr/bin/false",), password="rasp"))
