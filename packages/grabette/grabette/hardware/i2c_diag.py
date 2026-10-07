"""Why the gripper angle sensors cannot be reached, and what fixes it.

The dashboard's "Diagnose" button. A grabette whose AS5600L encoders do not come
up refuses to record ("No device found for /dev/i2c-3"), and the causes sit at
very different layers: the boot config never asked for the I2C buses, the boot
config asks but the Pi was not rebooted, the i2c-dev module that creates the
/dev nodes is not loaded, or the buses are there and a sensor simply does not
answer (wiring). Each has its own fix, and the error message cannot tell them
apart — this walks the layers bottom-up, stops at the first broken one, and
names the fix for it.

Pure reads, no root: everything here is a file under /boot, /sys, /proc, /etc or
/dev, plus one I2C read per sensor. The fixes themselves need root and run in
the API router (routers/angle.py) through narrow sudoers grants.

Paths are resolved under `root` so the whole walk can be tested against a fake
filesystem.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

# The two V2 sensor buses (see hardware/angle.py) and the overlay line that
# creates each one. Ordered distal, proximal — the order AngleCapture opens them.
SENSOR_BUSES = (
    ("distal", 3, "dtoverlay=i2c3,pins_4_5", "GPIO 4 (SDA) / GPIO 5 (SCL)"),
    ("proximal", 4, "dtoverlay=i2c4,pins_8_9", "GPIO 8 (SDA) / GPIO 9 (SCL)"),
)
AS5600L_ADDR = 0x40
AS5600_ADDR = 0x36  # the non-L part: same chip, other address — the wrong sensor
STATUS_REG = 0x0B
STATUS_MD = 0x20  # magnet detected

BOOT_CONFIGS = ("boot/firmware/config.txt", "boot/config.txt")
MODULES_LOAD_CONF = "/etc/modules-load.d/i2c-dev.conf"

# Fix ids — the vocabulary shared with routers/angle.py and the dashboard.
FIX_ENABLE_OVERLAYS = "enable_overlays"
FIX_REBOOT = "reboot"
FIX_LOAD_I2C_DEV = "load_i2c_dev"
FIX_PERSIST_I2C_DEV = "persist_i2c_dev"
FIX_REINIT = "reinit_sensors"

FIX_LABELS = {
    FIX_ENABLE_OVERLAYS: "Enable the I2C buses",
    FIX_REBOOT: "Reboot the grabette",
    FIX_LOAD_I2C_DEV: "Load the I2C driver",
    FIX_PERSIST_I2C_DEV: "Load the I2C driver at every boot",
    FIX_REINIT: "Reconnect the sensors",
}

OK, FAIL, WARN, SKIP = "ok", "fail", "warn", "skip"


@dataclass
class Check:
    key: str
    label: str
    status: str  # OK / FAIL / WARN / SKIP
    detail: str = ""


@dataclass
class Diagnosis:
    checks: list[Check] = field(default_factory=list)
    # One line saying what is wrong (or that nothing is).
    summary: str = ""
    # What the dashboard's fix button does (one of FIX_*), or None when there is
    # nothing the device can do for itself (a sensor that does not answer is a
    # cable, not a command).
    fix: str | None = None
    fix_label: str = ""
    # The same fix, by hand over SSH — shown whatever happens, and the only way
    # out when the service user is not allowed to run it (no sudoers grant).
    manual: list[str] = field(default_factory=list)
    healthy: bool = False
    # The sensor ("distal" / "proximal") whose cable to check, when one does not
    # answer at all: the dashboard shows where that cable runs.
    cable: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# A probe reads a register: (bus, addr, reg) -> byte, raising OSError when
# nothing answers. Injected so tests never touch a real bus.
Probe = Callable[[int, int, int], int]


def i2c_probe(bus: int, addr: int, reg: int) -> int:
    from adafruit_extended_bus import ExtendedI2C

    i2c = ExtendedI2C(bus)
    try:
        buf = bytearray(1)
        i2c.writeto_then_readfrom(addr, bytes([reg]), buf)
        return buf[0]
    finally:
        i2c.deinit()


def find_boot_config(root: Path) -> Path | None:
    for rel in BOOT_CONFIGS:
        p = root / rel
        if p.is_file():
            return p
    return None


def _active_lines(text: str) -> list[str]:
    """config.txt lines that take effect: comments dropped, spaces squeezed."""
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip().replace(" ", "")
        if line:
            out.append(line)
    return out


def _overlay_present(lines: list[str], wanted: str) -> bool:
    """`dtoverlay=i2c3,pins_4_5` present, in whatever parameter order."""
    name, _, params = wanted.partition("=")[2].partition(",")
    for line in lines:
        if not line.startswith("dtoverlay="):
            continue
        got_name, _, got_params = line.partition("=")[2].partition(",")
        if got_name == name and params in got_params.split(","):
            return True
    return False


def missing_overlays(config_text: str) -> list[str]:
    lines = _active_lines(config_text)
    return [ov for _, _, ov, _ in SENSOR_BUSES if not _overlay_present(lines, ov)]


def overlay_append_text(missing: list[str]) -> str:
    """What the fix appends to config.txt. Under [all]: the file may end inside
    a [pi5]/[cm4] filter, and lines landing there would be silently ignored."""
    return ("\n[all]\n# Grabette gripper angle sensors (added by the dashboard)\n"
            + "".join(f"{ov}\n" for ov in missing))


def _i2c_dev_persistent(root: Path) -> bool:
    files = [root / "etc/modules"]
    for d in ("etc/modules-load.d", "usr/lib/modules-load.d", "lib/modules-load.d"):
        files += sorted((root / d).glob("*.conf")) if (root / d).is_dir() else []
    for f in files:
        try:
            text = f.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if re.fullmatch(r"i2c[-_]dev", line.split("#", 1)[0].strip()):
                return True
    return False


def _i2c_dev_builtin(root: Path) -> bool:
    """Loaded but not a module: compiled into the kernel, nothing to persist."""
    try:
        modules = (root / "proc/modules").read_text()
    except OSError:
        return False
    return not any(line.split(" ", 1)[0] == "i2c_dev"
                   for line in modules.splitlines())


def diagnose(root: Path = Path("/"), probe: Probe = i2c_probe,
             backend_error: str = "", user: str = "rasp") -> Diagnosis:
    """Walk the layers bottom-up and stop at the first one that is broken."""
    d = Diagnosis()
    cfg_path = find_boot_config(root)
    cfg_shown = "/" + str(cfg_path.relative_to(root)) if cfg_path else "config.txt"

    # 1 — the boot config asks for the two buses.
    if cfg_path is None:
        d.checks.append(Check("boot_config", "Boot configuration", FAIL,
                              "No /boot/firmware/config.txt — is this a Raspberry Pi?"))
        d.summary = "This does not look like a Raspberry Pi: there is no boot configuration."
        return d
    missing = missing_overlays(cfg_path.read_text(errors="replace"))
    if missing:
        d.checks.append(Check("boot_config", "Boot configuration", FAIL,
                              f"{cfg_shown} does not enable " + ", ".join(missing)))
        d.summary = ("The I2C buses of the angle sensors are not enabled in the "
                     "boot configuration.")
        d.fix = FIX_ENABLE_OVERLAYS
        lines = "\\n".join(missing)
        d.manual = [f"printf '\\n[all]\\n{lines}\\n' | sudo tee -a {cfg_shown}",
                    "sudo reboot"]
        return _finish(d)
    d.checks.append(Check("boot_config", "Boot configuration", OK,
                          f"{cfg_shown} enables the I2C buses 3 and 4"))

    # 2 — the kernel created them (the overlays were applied at boot).
    absent = [b for _, b, _, _ in SENSOR_BUSES
              if not (root / f"sys/bus/i2c/devices/i2c-{b}").exists()]
    if absent:
        d.checks.append(Check("kernel_buses", "I2C buses", FAIL,
                              "Bus " + ", ".join(map(str, absent))
                              + " not created by the kernel"))
        d.summary = ("The boot configuration is right but has not been applied "
                     "yet — the grabette needs a reboot.")
        d.fix = FIX_REBOOT
        d.manual = ["sudo reboot"]
        return _finish(d)
    d.checks.append(Check("kernel_buses", "I2C buses", OK,
                          "The kernel created the buses 3 and 4"))

    # 3 — the i2c-dev driver, which is what turns a bus into /dev/i2c-N.
    loaded = (root / "sys/class/i2c-dev").is_dir()
    persistent = loaded and (_i2c_dev_builtin(root) or _i2c_dev_persistent(root))
    if not loaded:
        d.checks.append(Check("i2c_dev", "I2C driver (i2c-dev)", FAIL,
                              "Not loaded: no /dev/i2c-* device files exist"))
        d.summary = ("The i2c-dev driver is not loaded, so the sensors' buses "
                     "have no /dev/i2c-* file to open.")
        d.fix = FIX_LOAD_I2C_DEV
        d.manual = ["sudo modprobe i2c-dev",
                    f"echo i2c-dev | sudo tee {MODULES_LOAD_CONF}",
                    "sudo systemctl restart grabette"]
        return _finish(d)
    d.checks.append(Check(
        "i2c_dev", "I2C driver (i2c-dev)", OK if persistent else WARN,
        "Loaded" if persistent else
        "Loaded now, but not set to load at boot — the fault comes back "
        "after the next reboot"))

    # 4 — the device files, and the right to open them.
    for name, bus, _, _ in SENSOR_BUSES:
        node = root / f"dev/i2c-{bus}"
        if not node.exists():
            d.checks.append(Check(f"node_{bus}", f"/dev/i2c-{bus}", FAIL,
                                  "Missing although the driver is loaded"))
            d.summary = f"/dev/i2c-{bus} is missing although everything under it is in place."
            d.fix = FIX_REBOOT
            d.manual = ["sudo reboot"]
            return _finish(d)
        if root == Path("/") and not os.access(node, os.R_OK | os.W_OK):
            d.checks.append(Check(f"node_{bus}", f"/dev/i2c-{bus}", FAIL,
                                  f"The '{user}' user may not open it"))
            d.summary = (f"The grabette service is not allowed to use /dev/i2c-{bus} "
                         "(its user is not in the i2c group).")
            d.manual = [f"sudo usermod -aG i2c {user}", "sudo reboot"]
            return _finish(d)

    # 5 — each sensor answers on its bus.
    # Every sensor gets its row; the verdict names the first fault only — they
    # are fixed one at a time, the live refresh moving on to the next.
    first = None
    for name, bus, _, pins in SENSOR_BUSES:
        label = f"{name.capitalize()} sensor (bus {bus})"
        try:
            status = probe(bus, AS5600L_ADDR, STATUS_REG)
        except OSError:
            try:
                probe(bus, AS5600_ADDR, STATUS_REG)
                d.checks.append(Check(f"sensor_{bus}", label, FAIL,
                                      "An AS5600 answers at 0x36 — this grabette "
                                      "expects an AS5600L (0x40)"))
                first = first or (f"The {name} angle sensor is an AS5600: "
                                  "this grabette expects an AS5600L.", None)
            except OSError:
                d.checks.append(Check(f"sensor_{bus}", label, FAIL,
                                      f"No answer at 0x40 — check its cable on {pins}"))
                first = first or (f"The {name} angle sensor does not answer.", name)
            continue
        if status & STATUS_MD:
            d.checks.append(Check(f"sensor_{bus}", label, OK, "Answers, magnet detected"))
        else:
            d.checks.append(Check(f"sensor_{bus}", label, WARN,
                                  "Answers, but sees no magnet — check the magnet "
                                  "on the joint"))
    if first:
        d.summary, d.cable = first
        return _finish(d)

    # Everything below the daemon works. What is left is the daemon itself
    # still holding the fault from its failed bring-up, or the persistence gap.
    if backend_error:
        d.summary = ("The sensors work now, but the grabette has not "
                     "reconnected to them yet.")
        d.fix = FIX_REINIT
        if not persistent:
            d.summary += " The I2C driver is also not set to load at boot."
        return _finish(d)
    if not persistent:
        d.summary = ("The sensors work, but the I2C driver will not be loaded "
                     "after the next reboot.")
        d.fix = FIX_PERSIST_I2C_DEV
        d.manual = [f"echo i2c-dev | sudo tee {MODULES_LOAD_CONF}"]
        return _finish(d)
    d.summary = "The angle sensors work."
    d.healthy = True
    return d


def _finish(d: Diagnosis) -> Diagnosis:
    d.fix_label = FIX_LABELS.get(d.fix, "") if d.fix else ""
    return d
