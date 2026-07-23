from __future__ import annotations

import asyncio
import os
import platform
import re
import socket
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/time")
async def system_time():
    """Current wall clock + NTP sync state.

    Used by the sync orchestrator's preflight check: every peer should
    have ntp_synchronized=True and a small absolute offset_us before the
    orchestrator commits to a multi-device episode.
    """
    out = {
        "now_utc": datetime.now(timezone.utc).isoformat(),
        "ntp_synchronized": None,  # unknown → orchestrator can decide
        "offset_us": None,
        "ntp_server": None,
    }
    # timedatectl is portable across systemd-timesyncd and chrony — both
    # populate the same `timedatectl show` properties.
    show = await _run(["timedatectl", "show"])
    if show is not None:
        for line in show.splitlines():
            if line.startswith("NTPSynchronized="):
                out["ntp_synchronized"] = line.split("=", 1)[1] == "yes"
    # Timesync detailed status — has the most recent offset to NTP source.
    ts = await _run(["timedatectl", "timesync-status"])
    if ts is not None:
        for line in ts.splitlines():
            line = line.strip()
            # Offset can be reported as "+835us", "+2.588ms", "-1.2s", etc.
            # We always normalise to integer microseconds.
            m = re.match(r"Offset:\s*([+-]?\d+(?:\.\d+)?)([um]?s)", line)
            if m:
                n, unit = float(m.group(1)), m.group(2)
                factor = {"us": 1, "ms": 1000, "s": 1_000_000}[unit]
                out["offset_us"] = int(round(n * factor))
            if line.startswith("Server:"):
                out["ntp_server"] = line.split(":", 1)[1].strip()
    return out


async def _run(cmd: list[str]) -> str | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=1.0)
        if proc.returncode != 0:
            return None
        return out.decode(errors="replace")
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

    return info


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
