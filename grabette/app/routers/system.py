from __future__ import annotations

import asyncio
import os
import platform
import socket
import subprocess

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/api/system", tags=["system"])


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
