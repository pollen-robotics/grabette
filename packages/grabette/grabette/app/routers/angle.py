"""Gripper angle sensors: their state, and the dashboard's "Diagnose" button.

GET  /status    — the chip on the dashboard (up, or the fault blocking capture).
POST /diagnose  — walks the I2C stack (hardware/i2c_diag.py) and names the fix.
POST /fix       — applies the fix the diagnosis named.

The fixes that touch the system run as root through sudo, never with a
password: `make install-i2c-fix` installs a root-owned script with a fixed set
of actions (scripts/grabette-i2c-fix) and grants the service user exactly those.
Without that grant the popup says to run it. Rebooting goes through the system
router's reboot (`make install-reboot`).
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
from grabette.app.routers import system
from grabette.backend.base import Backend
from grabette.hardware import i2c_diag

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/angle", tags=["angle"])

# Installed by `make install-i2c-fix`, root-owned; each action is granted
# separately in sudoers, with its exact argument.
_HELPER = "/usr/local/sbin/grabette-i2c-fix"


class FixRequest(BaseModel):
    fix: str


class NotPermitted(Exception):
    """sudo refused: the grants of `make install-i2c-fix` are not installed."""


async def _helper(action: str) -> str:
    """Run one action of the root helper, return what it printed, or raise
    NotPermitted (sudo refused) / RuntimeError (the action failed)."""
    proc = await asyncio.create_subprocess_exec(
        "sudo", "-n", _HELPER, action,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0:
        text = err.decode(errors="replace").strip()
        # sudo's own refusals ("a password is required", "not allowed to
        # execute") start with its name; the helper's errors do not.
        if text.startswith("sudo:"):
            raise NotPermitted(text)
        raise RuntimeError(f"{action} failed: {text}")
    return out.decode(errors="replace").strip()


def _wait_for_nodes(timeout_s: float = 3.0) -> None:
    """udev creates /dev/i2c-N shortly after modprobe returns."""
    deadline = time.monotonic() + timeout_s
    nodes = [Path(f"/dev/i2c-{b}") for _, b, _, _ in i2c_diag.SENSOR_BUSES]
    while time.monotonic() < deadline and not all(n.exists() for n in nodes):
        time.sleep(0.1)


# What each fix runs as root, and the make target whose sudoers drop-in grants
# it. A fix missing here (reconnecting the sensors) needs no root.
_FIX_GRANTS = {
    i2c_diag.FIX_ENABLE_OVERLAYS: ([(_HELPER, "enable-overlays")],
                                   "install-i2c"),
    i2c_diag.FIX_LOAD_I2C_DEV: ([(_HELPER, "load"), (_HELPER, "persist")],
                                "install-i2c"),
    i2c_diag.FIX_PERSIST_I2C_DEV: ([(_HELPER, "persist")], "install-i2c"),
    i2c_diag.FIX_REBOOT: ([system.REBOOT_CMD], "install-reboot"),
}

# The open popup runs the diagnosis every 2 s, and sudo logs each listing:
# reuse it for a while. Long enough to keep the journal quiet, short enough
# that a `make install-i2c` just run shows up on the next walks.
_LISTING_TTL_S = 15.0
_listing_cache: tuple[float, str] | None = None


def _sudo_listing() -> str:
    global _listing_cache
    now = time.monotonic()
    if _listing_cache is None or now - _listing_cache[0] > _LISTING_TTL_S:
        _listing_cache = (now, system.sudo_listing())
    return _listing_cache[1]


def _without_ungranted_fix(diag: dict, listing: str) -> dict:
    """No fix button the device is not allowed to apply: a click would only
    fail on sudo. The diagnosis says which make target grants it instead, and
    puts it first in the by-hand commands (unless they already start with it)."""
    grant = _FIX_GRANTS.get(diag.get("fix"))
    if grant is None:
        return diag
    cmds, target = grant
    if all(system.sudo_grants_without_password(listing, c) for c in cmds):
        return diag
    make = i2c_diag.make_cmd(target)
    manual = diag.get("manual", [])
    if make not in manual:
        manual = [make, "# then run the diagnostic again — or, by hand:",
                  *manual]
    return {**diag, "fix": None, "fix_label": "",
            "summary": (f"{diag['summary']} The dashboard is not allowed to "
                        f"fix it: run `make {target}` on the grabette once, "
                        "or fix it by hand."),
            "manual": manual}


def _diagnose(backend: Backend) -> dict:
    status = backend.angle_sensors_status
    diag = i2c_diag.diagnose(backend_error=status.get("error", ""),
                             user=getpass.getuser()).to_dict()
    if diag.get("fix") in _FIX_GRANTS:
        diag = _without_ungranted_fix(diag, _sudo_listing())
    return diag


@router.get("/status")
def angle_status(backend: Backend = Depends(get_backend)):
    return backend.angle_sensors_status


@router.post("/diagnose")
async def angle_diagnose(backend: Backend = Depends(get_backend)):
    # The sensor probe is a real I2C read: never alongside the capture thread.
    system.refuse_while_recording("running the diagnostic", backend)
    return await asyncio.get_running_loop().run_in_executor(
        None, _diagnose, backend)


@router.post("/fix")
async def angle_fix(req: FixRequest, backend: Backend = Depends(get_backend)):
    """Apply one fix. Returns {"done": <what happened>, "reboot": bool,
    "diagnosis": <the walk run again>} so the popup shows the result at once."""
    system.refuse_while_recording("applying a fix", backend)
    loop = asyncio.get_running_loop()
    reboot = False
    try:
        if req.fix == i2c_diag.FIX_LOAD_I2C_DEV:
            await _helper("load")
            await _helper("persist")
            await loop.run_in_executor(None, _wait_for_nodes)
            await loop.run_in_executor(None, backend.reinit_angle_sensors)
            done = "The I2C driver is loaded, and will be at every boot."
        elif req.fix == i2c_diag.FIX_PERSIST_I2C_DEV:
            await _helper("persist")
            done = "The I2C driver will be loaded at every boot."
        elif req.fix == i2c_diag.FIX_REINIT:
            await loop.run_in_executor(None, backend.reinit_angle_sensors)
            done = "The grabette reconnected to the sensors."
        elif req.fix == i2c_diag.FIX_ENABLE_OVERLAYS:
            cfg = Path(await _helper("enable-overlays"))
            done = (f"The I2C buses are enabled in {cfg} (previous version kept "
                    f"as {cfg.name}.grabette-bak). Reboot to apply.")
            reboot = True
        elif req.fix == i2c_diag.FIX_REBOOT:
            await system.system_reboot()
            return {"done": "The grabette is rebooting — reload this page in "
                            "a minute.", "reboot": False, "rebooting": True,
                    "diagnosis": None}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown fix: {req.fix}")
    except NotPermitted as e:
        logger.warning("Angle sensors fix %s refused by sudo: %s", req.fix, e)
        raise HTTPException(
            status_code=500,
            detail="This fix is not permitted. Run `make install-i2c` on "
                   "the device.")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("Angle sensors fix %s applied", req.fix)
    diagnosis = None if reboot else await loop.run_in_executor(
        None, _diagnose, backend)
    return {"done": done, "reboot": reboot, "rebooting": False,
            "diagnosis": diagnosis}
