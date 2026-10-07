"""Gripper angle-sensor calibration, for the dashboard's "Calibrate my device".

The same zeroing scripts/calibrate_angles.py does, run by the daemon on its own
sensor handles: the fingers are opened fully by hand, then POST takes that pose
as zero, saves it and applies it — no restart, and the "not calibrated" fault
that refuses recording clears at once.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from grabette.app.dependencies import get_backend
from grabette.backend.base import Backend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/calibration", tags=["calibration"])


@router.get("")
def calibration_status(backend: Backend = Depends(get_backend)):
    return {"needs_calibration": backend.needs_calibration}


@router.post("")
async def calibrate(backend: Backend = Depends(get_backend)):
    try:
        calibration = await backend.calibrate_angles()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:  # noqa: BLE001 — an I2C error, reported as such
        logger.exception("Angle calibration failed")
        raise HTTPException(status_code=500,
                            detail=f"Calibration failed: {e or type(e).__name__}")
    return {"needs_calibration": backend.needs_calibration,
            "calibration": calibration}
