"""The angle-sensor diagnostic names the first broken layer, and its fix.

"No device found for /dev/i2c-3" has four very different causes — buses not
enabled in config.txt, enabled but not rebooted, i2c-dev not loaded, a sensor
that does not answer — and each wants a different fix. Every scenario below is
one a grabette has actually been in, laid out as a fake root filesystem.
"""
from pathlib import Path

import pytest

from grabette.hardware import i2c_diag as d

_CONFIG = """
dtparam=i2c_arm=on
#dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=6,i2c_gpio_scl=5
dtoverlay=i2c3,pins_4_5
dtoverlay=i2c4,pins_8_9
"""


def _touch(root: Path, rel: str, text: str = "") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _healthy_root(root: Path) -> Path:
    _touch(root, "boot/firmware/config.txt", _CONFIG)
    for b in (3, 4):
        (root / f"sys/bus/i2c/devices/i2c-{b}").mkdir(parents=True)
        _touch(root, f"dev/i2c-{b}")
    (root / "sys/class/i2c-dev").mkdir(parents=True)
    _touch(root, "proc/modules", "i2c_dev 16384 0 - Live 0x0\n")
    _touch(root, "etc/modules-load.d/i2c-dev.conf", "i2c-dev\n")
    return root


def _probe(answers: dict[tuple[int, int], int]):
    def probe(bus, addr, reg):
        if (bus, addr) not in answers:
            raise OSError(121, "Remote I/O error")
        return answers[(bus, addr)]
    return probe


_BOTH_OK = _probe({(3, 0x40): 0x20, (4, 0x40): 0x20})


def test_healthy_device(tmp_path):
    diag = d.diagnose(_healthy_root(tmp_path), _BOTH_OK)
    assert diag.healthy and diag.fix is None
    assert all(c.status == d.OK for c in diag.checks)


def test_overlays_missing_from_boot_config(tmp_path):
    root = _healthy_root(tmp_path)
    _touch(root, "boot/firmware/config.txt",
           "dtparam=i2c_arm=on\n#dtoverlay=i2c3,pins_4_5\ndtoverlay=i2c4,pins_8_9\n")
    diag = d.diagnose(root, _BOTH_OK)
    assert diag.fix == d.FIX_ENABLE_OVERLAYS
    assert "dtoverlay=i2c3,pins_4_5" in diag.checks[0].detail
    assert "i2c4" not in diag.checks[0].detail


def test_append_text_lands_under_all_and_adds_only_what_is_missing():
    text = d.overlay_append_text(d.missing_overlays("[pi5]\ndtoverlay=i2c4,pins_8_9\n"))
    assert text.startswith("\n[all]\n")
    assert "dtoverlay=i2c3,pins_4_5\n" in text and "i2c4" not in text


def test_overlay_params_in_any_order():
    assert d.missing_overlays("dtoverlay=i2c3,baudrate=100000,pins_4_5\n"
                              "dtoverlay = i2c4,pins_8_9\n") == []


def test_config_set_but_not_rebooted(tmp_path):
    root = _healthy_root(tmp_path)
    (root / "sys/bus/i2c/devices/i2c-4").rmdir()
    diag = d.diagnose(root, _BOTH_OK)
    assert diag.fix == d.FIX_REBOOT


def test_i2c_dev_not_loaded(tmp_path):
    """The case that started it all: buses exist, /dev/i2c-* do not."""
    root = _healthy_root(tmp_path)
    (root / "sys/class/i2c-dev").rmdir()
    for b in (3, 4):
        (root / f"dev/i2c-{b}").unlink()
    diag = d.diagnose(root, _BOTH_OK)
    assert diag.fix == d.FIX_LOAD_I2C_DEV
    assert "sudo modprobe i2c-dev" in diag.manual


def test_loaded_by_hand_but_not_at_boot(tmp_path):
    root = _healthy_root(tmp_path)
    (root / "etc/modules-load.d/i2c-dev.conf").unlink()
    diag = d.diagnose(root, _BOTH_OK)
    assert diag.fix == d.FIX_PERSIST_I2C_DEV
    assert not diag.healthy


def test_i2c_dev_in_etc_modules_counts(tmp_path):
    root = _healthy_root(tmp_path)
    (root / "etc/modules-load.d/i2c-dev.conf").unlink()
    _touch(root, "etc/modules", "# comment\ni2c-dev\n")
    assert d.diagnose(root, _BOTH_OK).healthy


def test_builtin_i2c_dev_needs_no_persistence(tmp_path):
    root = _healthy_root(tmp_path)
    (root / "etc/modules-load.d/i2c-dev.conf").unlink()
    _touch(root, "proc/modules", "snd 1 0 - Live 0x0\n")
    assert d.diagnose(root, _BOTH_OK).healthy


@pytest.mark.parametrize("answers, detail", [
    ({(3, 0x40): 0x20}, "No answer at 0x40"),
    ({(3, 0x40): 0x20, (4, 0x36): 0x20}, "AS5600 answers at 0x36"),
])
def test_sensor_not_answering_has_no_automatic_fix(tmp_path, answers, detail):
    diag = d.diagnose(_healthy_root(tmp_path), _probe(answers))
    assert diag.fix is None and not diag.healthy
    assert "proximal" in diag.summary
    assert detail in diag.checks[-1].detail


def test_missing_magnet_is_a_warning_not_a_failure(tmp_path):
    diag = d.diagnose(_healthy_root(tmp_path),
                      _probe({(3, 0x40): 0x00, (4, 0x40): 0x20}))
    assert diag.healthy
    assert diag.checks[-2].status == d.WARN


def test_sensors_fine_but_daemon_still_faulted(tmp_path):
    diag = d.diagnose(_healthy_root(tmp_path), _BOTH_OK,
                      backend_error="could not be initialised")
    assert diag.fix == d.FIX_REINIT


def test_not_a_pi(tmp_path):
    diag = d.diagnose(tmp_path, _BOTH_OK)
    assert diag.fix is None and not diag.healthy


# --- the dashboard side -------------------------------------------------------

def test_angle_chip_state():
    from grabette.ui.app import _angle_state

    assert _angle_state(None) == "unknown"
    assert _angle_state({"enabled": False}) is None  # no chip at all
    assert _angle_state({"enabled": True, "initialized": True, "error": ""}) == "connected"
    assert _angle_state({"enabled": True, "initialized": False,
                         "error": "could not be initialised"}) == "missing"


def test_report_shows_checks_verdict_and_manual_commands(tmp_path):
    from grabette.ui.app import _diag_report_html

    root = _healthy_root(tmp_path)
    (root / "sys/class/i2c-dev").rmdir()
    out = _diag_report_html(d.diagnose(root, _BOTH_OK).to_dict())
    assert "I2C driver (i2c-dev)" in out
    assert "i2c-dev driver is not loaded" in out
    assert "sudo modprobe i2c-dev" in out


def test_report_shows_the_cable_guide_of_the_silent_sensor(tmp_path):
    from grabette.ui.app import _diag_report_html

    root = _healthy_root(tmp_path)

    def report(answers):
        return _diag_report_html(d.diagnose(root, _probe(answers)).to_dict())

    assert "proximal-sensor-cable.gif" in report({(3, 0x40): 0x20})
    assert "distal-sensor-cable.gif" in report({(4, 0x40): 0x20})
    assert "sensor-cable.gif" not in report({(3, 0x40): 0x20, (4, 0x40): 0x20})
    # A wrong chip is not a cable.
    assert "sensor-cable.gif" not in report({(3, 0x40): 0x20, (4, 0x36): 0x20})


def test_verdict_names_one_sensor_fault_at_a_time(tmp_path):
    root = _healthy_root(tmp_path)
    no_proximal = d.diagnose(root, _probe({(3, 0x40): 0x20}))
    assert no_proximal.summary == "The proximal angle sensor does not answer."
    assert no_proximal.cable == "proximal"
    # Both down: both rows fail, but the verdict and the guide are the distal
    # one's; the proximal one's comes once the distal sensor answers.
    both = d.diagnose(root, _probe({}))
    assert [c.status for c in both.checks[-2:]] == [d.FAIL, d.FAIL]
    assert both.summary == "The distal angle sensor does not answer."
    assert both.cable == "distal"
    wrong_chip = d.diagnose(root, _probe({(3, 0x40): 0x20, (4, 0x36): 0x20}))
    assert "is an AS5600" in wrong_chip.summary and wrong_chip.cable is None
