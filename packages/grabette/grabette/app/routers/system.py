from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import socket
import time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])

# Exact command the sudoers drop-in (see `make install-poweroff`) grants the
# unprivileged service user. The pre-check, the permission rule, and the actual
# dispatch must all use this identical string or the NOPASSWD match fails.
_POWEROFF_CMD = ("/usr/bin/systemctl", "poweroff")
# Same, for `make install-reboot`.
_REBOOT_CMD = ("/usr/bin/systemctl", "reboot")

# A runas list then the commands of one `sudo -l` entry: "(root) NOPASSWD: /x y".
_SUDO_ENTRY_RE = re.compile(r"^\s*\(([^)]*)\)\s*(.*)$")
# One leading tag of a command: "NOPASSWD: ", "SETENV: "…
_SUDO_TAG_RE = re.compile(r"^([A-Z_]+):\s*(.*)$")

# Keeps the pending dispatches referenced: the loop holds tasks weakly.
_pending_sudo: set[asyncio.Task] = set()


def sudo_grants_without_password(listing: str, cmd: tuple[str, ...]) -> bool:
    """Whether a `sudo -l` listing lets this user run ``cmd`` as root without
    a password: a NOPASSWD entry for ALL, for the exact command, or for the
    program with any arguments. A tag holds for the commands after it in its
    entry, until PASSWD: undoes it."""
    full, program = " ".join(cmd), cmd[0]
    for line in listing.splitlines():
        entry = _SUDO_ENTRY_RE.match(line)
        if not entry:
            continue
        users = entry.group(1).split(":")[0]
        if not {u.strip() for u in users.split(",")} & {"root", "ALL"}:
            continue
        nopasswd = False
        for item in entry.group(2).split(","):
            item = item.strip()
            while tag := _SUDO_TAG_RE.match(item):
                if tag.group(1) in ("NOPASSWD", "PASSWD"):
                    nopasswd = tag.group(1) == "NOPASSWD"
                item = tag.group(2)
            if nopasswd and item in ("ALL", full, program):
                return True
    return False


async def sudo_without_password(cmd: tuple[str, ...]) -> bool:
    """Whether the service user may run ``cmd`` as root with no password.

    Not `sudo -n -l <cmd>`: it answers whether the command is *allowed*, so a
    user in the sudo group passes it, and the `sudo -n` that follows then
    fails on the password. The whole listing names the NOPASSWD grants; it is
    readable without a password as soon as the user holds one (listpw=any),
    and when it is not, there is no grant to find."""
    proc = await asyncio.create_subprocess_exec(
        "sudo", "-n", "-l",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    return proc.returncode == 0 and sudo_grants_without_password(
        out.decode(errors="replace"), cmd)


async def _sudo_after(cmd: tuple[str, ...], delay_s: float) -> None:
    await asyncio.sleep(delay_s)
    proc = await asyncio.create_subprocess_exec(
        "sudo", "-n", *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        # The caller was already told it worked: the journal is all that's left.
        logger.error("%s failed: %s", " ".join(cmd),
                     err.decode(errors="replace").strip())


def sudo_later(cmd: tuple[str, ...], delay_s: float = 1.0) -> None:
    """Run ``cmd`` through `sudo -n` a little later, so the reply announcing it
    reaches the browser (or the fleet) before systemd stops this service."""
    task = asyncio.create_task(_sudo_after(cmd, delay_s))
    _pending_sudo.add(task)
    task.add_done_callback(_pending_sudo.discard)


def refuse_while_recording(action: str) -> None:
    """409 mid-recording: powering off or rebooting would truncate the
    episode. If the daemon isn't running (mock backend / not yet initialised)
    there's nothing to protect — allow it."""
    try:
        from grabette.app.main import get_daemon_instance

        daemon = get_daemon_instance()
        if daemon is not None and daemon.state.value == "running":
            cap = daemon.backend.get_state().capture
            if cap.is_capturing or cap.is_starting:
                raise HTTPException(
                    status_code=409,
                    detail=f"A recording is in progress. Stop the capture "
                           f"before {action}.",
                )
    except HTTPException:
        raise
    except Exception:
        pass

# PiSugar 3 power-management registers (I2C 0x57). Confirmed against the
# pisugar-power-manager-rs driver and validated on-device: the P3 has no
# force-shutdown opcode — cutting power means clearing the output-enable bit
# (bit 5) of control register 0x02. Protected registers must be unlocked by
# writing 0x29 to 0x0B *before each* write (the firmware re-locks after every
# one). Register 0x09 is a countdown in seconds after which the rail is cut;
# its timing is loose and tends to undershoot, so keep the value generous.
_PISUGAR_ADDR = 0x57
_PISUGAR_WRITE_ENABLE = 0x0B
_PISUGAR_UNLOCK = 0x29
_PISUGAR_COUNTDOWN = 0x09
_PISUGAR_CTRL1 = 0x02
_PISUGAR_OUTPUT_BIT = 1 << 5
_PISUGAR_POWER_PLUGGED_BIT = 1 << 7  # CTRL1 bit 7: external power present (VBUS)


_battery_ema: float | None = None

# The battery % is slow-moving (and already median+EMA smoothed), so we only
# hit the I2C fuel gauge every _BATTERY_READ_INTERVAL_S and serve a cached value
# in between — /api/system/info is polled far more often than that (e.g. the
# Episodes status strip). Charging state is read separately and stays live so
# the low-battery alert still clears the instant Grabette is plugged in.
_BATTERY_READ_INTERVAL_S = 30.0
_battery_cached: float | None = None
_battery_last_read: float | None = None


def _pisugar_cut_power(delay_s: int = 30) -> bool:
    """Arm the PiSugar 3 UPS to physically cut its 5V rail after ``delay_s``.

    Without this, `systemctl poweroff` halts the OS but the battery-backed UPS
    keeps feeding the board, so it never actually powers down (the red PWR LED
    and the PiSugar's own LED stay lit). Arming the countdown here lets the OS
    halt cleanly first, then the PiSugar removes power for a true power-off.

    Best-effort: on a device with no PiSugar (or if I2C is unavailable) this
    returns False and the caller still proceeds with the OS poweroff.
    """
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        try:
            # Set the shutdown countdown (protected register).
            bus.write_byte_data(_PISUGAR_ADDR, _PISUGAR_WRITE_ENABLE, _PISUGAR_UNLOCK)
            bus.write_byte_data(_PISUGAR_ADDR, _PISUGAR_COUNTDOWN, delay_s & 0xFF)
            # Clear the output-enable bit to start the countdown to cut power.
            ctrl1 = bus.read_byte_data(_PISUGAR_ADDR, _PISUGAR_CTRL1)
            bus.write_byte_data(_PISUGAR_ADDR, _PISUGAR_WRITE_ENABLE, _PISUGAR_UNLOCK)
            bus.write_byte_data(_PISUGAR_ADDR, _PISUGAR_CTRL1, ctrl1 & ~_PISUGAR_OUTPUT_BIT)
        finally:
            bus.close()
        return True
    except Exception:
        return False


def _pisugar_battery() -> float | None:
    """Read battery percentage from PiSugar 3 via I2C (addr 0x57, reg 0x2A).

    Takes the median of 3 rapid reads to discard I2C glitches, then applies
    an EMA (α=0.2) to prevent the displayed percentage from bouncing up and
    down due to fuel-gauge noise or transient load changes.

    Throttled: the actual I2C read runs at most once per
    _BATTERY_READ_INTERVAL_S; more frequent calls return the cached value.
    """
    global _battery_ema, _battery_cached, _battery_last_read

    now = time.monotonic()
    if (
        _battery_cached is not None
        and _battery_last_read is not None
        and now - _battery_last_read < _BATTERY_READ_INTERVAL_S
    ):
        return _battery_cached

    try:
        import smbus2
        bus = smbus2.SMBus(1)
        readings = sorted(bus.read_byte_data(0x57, 0x2A) for _ in range(3))
        bus.close()
        sample = float(readings[1])  # median of 3
    except Exception:
        # Read failed — keep the previous value (if any) rather than blinking to
        # N/A, and don't arm the throttle so the next call retries immediately.
        return _battery_cached

    if _battery_ema is None:
        _battery_ema = sample
    else:
        _battery_ema = 0.2 * sample + 0.8 * _battery_ema

    _battery_cached = round(_battery_ema)
    _battery_last_read = now
    return _battery_cached


def _pisugar_charging() -> bool | None:
    """Whether external power is plugged into the PiSugar 3 (CTRL1 0x02, bit 7).

    We report the "power plugged" bit rather than the stricter charging state
    (bit 7 AND the allow-charging bit 6) so the low-battery warning clears the
    instant Grabette is reconnected — even if, near full charge, the pack has
    momentarily stopped drawing current. Below the warning threshold a plugged
    pack is always actively charging, so the two are equivalent there.

    Returns None when there is no PiSugar / I2C is unavailable.
    """
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        try:
            ctrl1 = bus.read_byte_data(_PISUGAR_ADDR, _PISUGAR_CTRL1)
        finally:
            bus.close()
        return bool(ctrl1 & _PISUGAR_POWER_PLUGGED_BIT)
    except Exception:
        return None


@router.get("/info")
def system_info():
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.machine(),
        "python": platform.python_version(),
    }

    # IP address
    from grabette.wifi import get_route_ip

    info["ip"] = get_route_ip() or "unknown"

    # Disk usage
    try:
        stat = os.statvfs("/")
        total = stat.f_frsize * stat.f_blocks
        free = stat.f_frsize * stat.f_bavail
        info["disk_total_gb"] = round(total / (1024**3), 1)
        info["disk_free_gb"] = round(free / (1024**3), 1)
    except Exception:
        pass

    # CPU temperature (RPi)
    try:
        temp = open("/sys/class/thermal/thermal_zone0/temp").read().strip()
        info["cpu_temp_c"] = round(int(temp) / 1000, 1)
    except Exception:
        pass

    # PiSugar battery
    battery = _pisugar_battery()
    if battery is not None:
        info["battery_pct"] = battery
        charging = _pisugar_charging()
        if charging is not None:
            info["battery_charging"] = charging

    return info


@router.websocket("/logs/ws")
async def stream_logs(ws: WebSocket):
    """Stream journalctl logs for the grabette service."""
    await ws.accept()
    try:
        proc = await asyncio.create_subprocess_exec(
            "journalctl", "-u", "grabette", "-f", "-n", "50", "--no-pager",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            await ws.send_text(line.decode(errors="replace").rstrip())
    except WebSocketDisconnect:
        pass
    except FileNotFoundError:
        await ws.send_text("journalctl not available")
    finally:
        try:
            proc.kill()
        except Exception:
            pass


@router.post("/shutdown")
async def system_shutdown():
    """Cleanly power off the Raspberry Pi.

    Refuses while a recording is in progress (would truncate the episode), then
    verifies the service user is actually allowed to run the poweroff command
    before dispatching it. The poweroff itself is fired detached with a short
    delay so this HTTP response reaches the browser before systemd tears the
    server down.
    """
    refuse_while_recording("powering off")
    if not await sudo_without_password(_POWEROFF_CMD):
        raise HTTPException(
            status_code=500,
            detail="Shutdown not permitted. Run 'make install-poweroff' on the device.",
        )

    # On a battery-backed device, `systemctl poweroff` only halts the OS — the
    # PiSugar UPS keeps the board powered. Arm the UPS to cut its 5V rail after
    # a short delay so the OS finishes halting first, then power actually drops.
    # No-op (harmless) on devices without a PiSugar.
    _pisugar_cut_power(30)

    sudo_later(_POWEROFF_CMD)
    return {"status": "shutting_down"}


@router.post("/reboot")
async def system_reboot():
    """Reboot the Raspberry Pi — the dashboard's way to make the firmware look
    for an RGB camera plugged back in, or to apply a boot config change.

    Same guards and dispatch as the shutdown, with its own sudoers grant
    (`make install-reboot`): never a password asked in the browser.
    """
    refuse_while_recording("rebooting")
    if not await sudo_without_password(_REBOOT_CMD):
        raise HTTPException(
            status_code=500,
            detail="Reboot not permitted. Run 'make install-reboot' on the device.",
        )
    sudo_later(_REBOOT_CMD)
    return {"status": "rebooting"}


@router.post("/update")
async def system_update():
    """Pull latest code from git and signal restart."""
    try:
        result = await asyncio.create_subprocess_exec(
            "git", "pull", "--ff-only",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "output": stdout.decode(errors="replace").strip(),
            "error": stderr.decode(errors="replace").strip() if result.returncode != 0 else None,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
