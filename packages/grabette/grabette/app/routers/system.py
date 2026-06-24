from __future__ import annotations

import asyncio
import os
import platform
import socket
import subprocess

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/api/system", tags=["system"])

# Exact command the sudoers drop-in (see `make install-poweroff`) grants the
# unprivileged service user. The pre-check, the permission rule, and the actual
# dispatch must all use this identical string or the NOPASSWD match fails.
_POWEROFF_CMD = ("/usr/bin/systemctl", "poweroff")


_battery_ema: float | None = None


def _pisugar_battery() -> float | None:
    """Read battery percentage from PiSugar 3 via I2C (addr 0x57, reg 0x2A).

    Takes the median of 3 rapid reads to discard I2C glitches, then applies
    an EMA (α=0.2) to prevent the displayed percentage from bouncing up and
    down due to fuel-gauge noise or transient load changes.
    """
    global _battery_ema
    try:
        import smbus2
        bus = smbus2.SMBus(1)
        readings = sorted(bus.read_byte_data(0x57, 0x2A) for _ in range(3))
        bus.close()
        sample = float(readings[1])  # median of 3
    except Exception:
        return None

    if _battery_ema is None:
        _battery_ema = sample
    else:
        _battery_ema = 0.2 * sample + 0.8 * _battery_ema

    return round(_battery_ema)


@router.get("/info")
def system_info():
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.machine(),
        "python": platform.python_version(),
    }

    # IP address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info["ip"] = s.getsockname()[0]
        s.close()
    except Exception:
        info["ip"] = "unknown"

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
    # Guard: never power off mid-recording. If the daemon isn't running (mock
    # backend / not yet initialised) there's nothing to protect — allow it.
    try:
        from grabette.app.main import get_daemon_instance

        daemon = get_daemon_instance()
        if daemon is not None and daemon.state.value == "running":
            cap = daemon.backend.get_state().capture
            if cap.is_capturing or cap.is_starting:
                raise HTTPException(
                    status_code=409,
                    detail="A recording is in progress. Stop the capture before powering off.",
                )
    except HTTPException:
        raise
    except Exception:
        pass

    # Permission pre-check: `sudo -n -l <cmd>` reports whether the command is
    # permitted without running it or prompting for a password.
    check = await asyncio.create_subprocess_exec(
        "sudo", "-n", "-l", *_POWEROFF_CMD,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await check.wait()
    if check.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail="Shutdown not permitted. Run 'make install-poweroff' on the device.",
        )

    # Dispatch detached so the 200 below is sent before the box goes down.
    await asyncio.create_subprocess_exec(
        "sh", "-c", f"sleep 1; sudo -n {' '.join(_POWEROFF_CMD)}",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return {"status": "shutting_down"}


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
