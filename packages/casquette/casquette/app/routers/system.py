from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import socket
import subprocess

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from casquette.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])

# Exact command the sudoers drop-in (`make install-poweroff`) grants the service
# user. The check, the rule and the dispatch must use this identical string.
_POWEROFF_CMD = ("/usr/bin/systemctl", "poweroff")

# A runas list then the commands of one `sudo -l` entry: "(root) NOPASSWD: /x y".
_SUDO_ENTRY_RE = re.compile(r"^\s*\(([^)]*)\)\s*(.*)$")
# One leading tag of a command: "NOPASSWD: ", "SETENV: "…
_SUDO_TAG_RE = re.compile(r"^([A-Z_]+):\s*(.*)$")

# Keeps the pending dispatch referenced: the loop holds tasks weakly.
_pending_sudo: set[asyncio.Task] = set()


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

    # Power: vcgencmd get_throttled → "0x0" when healthy.
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=2).stdout
        info["throttled"] = out.strip().split("=", 1)[1] if "=" in out else None
    except Exception:
        info["throttled"] = None

    # Memory: the Pi Zero 2W has 512 MB, and running short is what wedges the camera.
    try:
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                mem[key] = int(rest.split()[0])
        info["mem_total_mb"] = round(mem["MemTotal"] / 1024)
        info["mem_available_mb"] = round(mem["MemAvailable"] / 1024)
    except Exception:
        pass

    try:
        with open("/proc/uptime") as f:
            info["uptime_s"] = int(float(f.read().split()[0]))
    except Exception:
        pass

    info["device_id"] = settings.device_id
    info["device_name"] = settings.device_name
    info["relay_url"] = settings.relay_url
    return info


def sudo_grants_without_password(listing: str, cmd: tuple[str, ...]) -> bool:
    """Whether a `sudo -l` listing lets this user run ``cmd`` as root without a
    password: a NOPASSWD entry for ALL, the exact command, or the program with
    any arguments. (Same as grabette's system router.)"""
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


async def _sudo_without_password(cmd: tuple[str, ...]) -> bool:
    try:
        proc = await asyncio.create_subprocess_exec(
            "sudo", "-n", "-l",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False
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
        logger.error("%s failed: %s", " ".join(cmd), err.decode(errors="replace").strip())


@router.post("/shutdown")
async def system_shutdown():
    """Cleanly power off the Pi. Refused mid-recording (it would truncate the
    episode); fired after a short delay so the reply reaches the browser."""
    from casquette.app.main import get_daemon_instance

    daemon = get_daemon_instance()
    if daemon is not None and daemon.state.value == "running" and daemon.backend.is_capturing:
        raise HTTPException(status_code=409, detail="A recording is in progress")
    if not await _sudo_without_password(_POWEROFF_CMD):
        raise HTTPException(
            status_code=500,
            detail="Shutdown not permitted. Run 'make install-poweroff' on the device.",
        )
    task = asyncio.create_task(_sudo_after(_POWEROFF_CMD, 1.0))
    _pending_sudo.add(task)
    task.add_done_callback(_pending_sudo.discard)
    return {"status": "shutting_down"}


@router.websocket("/logs/ws")
async def stream_logs(ws: WebSocket):
    """Stream journalctl logs for the casquette service."""
    await ws.accept()
    try:
        proc = await asyncio.create_subprocess_exec(
            "journalctl", "-u", "casquette", "-f", "-n", "50", "--no-pager",
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
