"""Gripper angle sensors: their state, and the dashboard's "Diagnose" button.

GET  /status    — the chip on the dashboard (up, or the fault blocking capture).
POST /diagnose  — walks the I2C stack (hardware/i2c_diag.py) and names the fix.
POST /fix       — applies the fix the diagnosis named.

The fixes that touch the system run as root through `sudo -n`, each one an exact
command the sudoers drop-in of `make install-i2c-fix` grants the service user,
and nothing more. Without that grant they are refused with the commands to run
by hand — the diagnosis is still worth having.
"""

from __future__ import annotations

import asyncio
import getpass
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from grabette.app.dependencies import get_backend
from grabette.backend.base import Backend
from grabette.hardware import i2c_diag

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/angle", tags=["angle"])

# Exact commands granted by `make install-i2c-fix` — any difference in an
# argument and the NOPASSWD rule no longer matches.
_MODPROBE_CMD = ("/usr/sbin/modprobe", "i2c-dev")
_PERSIST_CMD = ("/usr/bin/tee", i2c_diag.MODULES_LOAD_CONF)
_REBOOT_CMD = ("/usr/bin/systemctl", "reboot")


def _backup_cmd(cfg: str) -> tuple[str, ...]:
    return ("/usr/bin/cp", cfg, f"{cfg}.grabette-bak")


def _append_cmd(cfg: str) -> tuple[str, ...]:
    return ("/usr/bin/tee", "-a", cfg)


class FixRequest(BaseModel):
    fix: str


def _refuse_while_recording(backend: Backend) -> None:
    cap = backend.get_capture_status()
    if cap.is_capturing or cap.is_starting:
        raise HTTPException(status_code=409,
                            detail="A recording is in progress — stop it first.")


async def _sudo(cmd: tuple[str, ...], stdin: str | None = None) -> None:
    """Run one granted command as root, or raise PermissionError/RuntimeError."""
    check = await asyncio.create_subprocess_exec(
        "sudo", "-n", "-l", *cmd,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    if await check.wait() != 0:
        raise PermissionError(" ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        "sudo", "-n", *cmd,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate(stdin.encode() if stdin is not None else None)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: "
                           f"{err.decode(errors='replace').strip()}")


def _wait_for_nodes(timeout_s: float = 3.0) -> None:
    """udev creates /dev/i2c-N shortly after modprobe returns."""
    deadline = time.monotonic() + timeout_s
    nodes = [Path(f"/dev/i2c-{b}") for _, b, _, _ in i2c_diag.SENSOR_BUSES]
    while time.monotonic() < deadline and not all(n.exists() for n in nodes):
        time.sleep(0.1)


def _diagnose(backend: Backend) -> dict:
    status = backend.angle_sensors_status
    return i2c_diag.diagnose(backend_error=status.get("error", ""),
                             user=getpass.getuser()).to_dict()


@router.get("/status")
def angle_status(backend: Backend = Depends(get_backend)):
    return backend.angle_sensors_status


@router.post("/diagnose")
async def angle_diagnose(backend: Backend = Depends(get_backend)):
    # The sensor probe is a real I2C read: never alongside the capture thread.
    _refuse_while_recording(backend)
    return await asyncio.get_running_loop().run_in_executor(
        None, _diagnose, backend)


@router.post("/fix")
async def angle_fix(req: FixRequest, backend: Backend = Depends(get_backend)):
    """Apply one fix. Returns {"done": <what happened>, "reboot": bool,
    "diagnosis": <the walk run again>} so the popup shows the result at once."""
    _refuse_while_recording(backend)
    loop = asyncio.get_running_loop()
    reboot = False
    try:
        if req.fix == i2c_diag.FIX_LOAD_I2C_DEV:
            await _sudo(_MODPROBE_CMD)
            await _sudo(_PERSIST_CMD, "i2c-dev\n")
            await loop.run_in_executor(None, _wait_for_nodes)
            await loop.run_in_executor(None, backend.reinit_angle_sensors)
            done = "The I2C driver is loaded, and will be at every boot."
        elif req.fix == i2c_diag.FIX_PERSIST_I2C_DEV:
            await _sudo(_PERSIST_CMD, "i2c-dev\n")
            done = "The I2C driver will be loaded at every boot."
        elif req.fix == i2c_diag.FIX_REINIT:
            await loop.run_in_executor(None, backend.reinit_angle_sensors)
            done = "The grabette reconnected to the sensors."
        elif req.fix == i2c_diag.FIX_ENABLE_OVERLAYS:
            cfg = i2c_diag.find_boot_config(Path("/"))
            if cfg is None:
                raise HTTPException(status_code=404,
                                    detail="No boot configuration found.")
            missing = i2c_diag.missing_overlays(cfg.read_text(errors="replace"))
            if missing:
                await _sudo(_backup_cmd(str(cfg)))
                await _sudo(_append_cmd(str(cfg)),
                            i2c_diag.overlay_append_text(missing))
            done = (f"The I2C buses are enabled in {cfg} (previous version kept "
                    f"as {cfg.name}.grabette-bak). Reboot to apply.")
            reboot = True
        elif req.fix == i2c_diag.FIX_REBOOT:
            # Checked here, dispatched detached below so this reply gets out.
            check = await asyncio.create_subprocess_exec(
                "sudo", "-n", "-l", *_REBOOT_CMD,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL)
            if await check.wait() != 0:
                raise PermissionError(" ".join(_REBOOT_CMD))
            await asyncio.create_subprocess_exec(
                "sh", "-c", f"sleep 1; sudo -n {' '.join(_REBOOT_CMD)}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL)
            return {"done": "The grabette is rebooting — reload this page in "
                            "a minute.", "reboot": False, "rebooting": True,
                    "diagnosis": None}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown fix: {req.fix}")
    except PermissionError as e:
        raise HTTPException(
            status_code=403,
            detail=(f"The grabette is not allowed to run `{e}` on its own. Run "
                    "the commands below over SSH, or `make install-i2c-fix` "
                    "once to enable this button."))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("Angle sensors fix %s applied", req.fix)
    diagnosis = None if reboot else await loop.run_in_executor(
        None, _diagnose, backend)
    return {"done": done, "reboot": reboot, "rebooting": False,
            "diagnosis": diagnosis}
