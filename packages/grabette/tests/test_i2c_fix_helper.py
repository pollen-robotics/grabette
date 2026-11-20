"""scripts/grabette-i2c-fix, the root side of the angle-sensor fixes: what it
writes, and that it agrees with hardware/i2c_diag.py on which overlays are
missing (the diagnosis names the fix, the helper applies it — a disagreement
would offer a fix that changes nothing, forever)."""
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

import pytest

from grabette.hardware import i2c_diag

SCRIPT = Path(__file__).parent.parent / "scripts" / "grabette-i2c-fix"


def _load():
    loader = SourceFileLoader("grabette_i2c_fix", str(SCRIPT))
    module = module_from_spec(spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


helper = _load()


def _run(*args):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


def _root(tmp_path, config: str | None):
    if config is not None:
        cfg = tmp_path / "boot/firmware/config.txt"
        cfg.parent.mkdir(parents=True)
        cfg.write_text(config)
    return tmp_path


def test_same_overlays_as_the_diagnosis():
    assert helper.OVERLAYS == tuple(ov for _, _, ov, _ in i2c_diag.SENSOR_BUSES)
    assert helper.BOOT_CONFIGS == i2c_diag.BOOT_CONFIGS
    assert "/" + helper.MODULES_LOAD_CONF == i2c_diag.MODULES_LOAD_CONF


@pytest.mark.parametrize("config", [
    "",
    "dtparam=i2c_arm=on\n",
    "dtoverlay=i2c3,pins_4_5\n",
    "dtoverlay = i2c3, pins_4_5\ndtoverlay=i2c4,pins_8_9\n",
    "dtoverlay=i2c3,baudrate=100000,pins_4_5\n",
    "#dtoverlay=i2c3,pins_4_5\ndtoverlay=i2c4,pins_8_9 # proximal\n",
    "dtoverlay=i2c3,pins_6_7\n",
])
def test_same_missing_overlays_as_the_diagnosis(config):
    assert helper.missing_overlays(config) == i2c_diag.missing_overlays(config)
    assert helper.append_text(["x"]) == i2c_diag.overlay_append_text(["x"])


def test_enable_overlays_appends_the_missing_ones_after_a_backup(tmp_path):
    original = "[pi5]\ndtoverlay=i2c3,pins_4_5\n"
    root = _root(tmp_path, original)
    res = _run("enable-overlays", root)
    assert res.returncode == 0, res.stderr
    cfg = root / "boot/firmware/config.txt"
    assert res.stdout.strip() == str(cfg)
    assert (root / "boot/firmware/config.txt.grabette-bak").read_text() == original
    # The [pi5] one does count as present (as for the diagnosis): only i2c4.
    assert cfg.read_text() == original + i2c_diag.overlay_append_text(
        ["dtoverlay=i2c4,pins_8_9"])
    # Run again: nothing left to add, nothing touched.
    assert _run("enable-overlays", root).returncode == 0
    assert i2c_diag.missing_overlays(cfg.read_text()) == []
    assert cfg.read_text().count("i2c4") == 1


def test_enable_overlays_without_a_boot_config(tmp_path):
    res = _run("enable-overlays", _root(tmp_path, None))
    assert res.returncode != 0
    assert "No boot configuration" in res.stderr


def test_persist_writes_the_module_name(tmp_path):
    assert _run("persist", tmp_path).returncode == 0
    assert (tmp_path / "etc/modules-load.d/i2c-dev.conf").read_text() == "i2c-dev\n"
    assert i2c_diag._i2c_dev_persistent(tmp_path)


def test_check_overlays_lists_the_missing_ones(tmp_path):
    root = _root(tmp_path, "dtoverlay=i2c3,pins_4_5\n")
    res = _run("check-overlays", root)
    assert res.returncode == 1
    assert res.stdout.split() == ["dtoverlay=i2c4,pins_8_9"]
    (root / "boot/firmware/config.txt").write_text(
        "dtoverlay=i2c3,pins_4_5\ndtoverlay=i2c4,pins_8_9\n")
    assert _run("check-overlays", root).returncode == 0


@pytest.mark.parametrize("args", [[], ["rm"], ["persist", "/", "extra"]])
def test_anything_else_is_refused(args):
    res = _run(*args)
    assert res.returncode != 0 and "usage" in res.stderr
