"""WiFi endpoints for the dashboard's Network page.

GET  /api/wifi/status         → current SSID, IP and signal
GET  /api/wifi/scan           → visible networks
POST /api/wifi/connect        → join a network (runs in the background, 202)
GET  /api/wifi/connect-result → outcome of the last attempt
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from casquette.fleet.wifi import (
    get_current_ssid,
    get_route_ip,
    get_signal_dbm,
    scan_networks,
    wifi_connect,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wifi", tags=["wifi"])


class ConnectRequest(BaseModel):
    ssid: str
    password: str = ""


_last_connect: dict = {"status": "idle", "message": ""}


@router.get("/status")
def wifi_status() -> dict:
    try:
        ssid = get_current_ssid()
    except (OSError, FileNotFoundError):  # no nmcli (dev machine)
        ssid = None
    return {"ssid": ssid, "ip": get_route_ip() or None, "signal_dbm": get_signal_dbm()}


@router.get("/scan")
def wifi_scan() -> list[dict]:
    try:
        return scan_networks()
    except (OSError, FileNotFoundError):
        return []


def _do_connect(ssid: str, password: str) -> None:
    global _last_connect
    try:
        result = wifi_connect(ssid, password)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[wifi] connect failed")
        result = f"ERROR: {exc}"
    logger.info("[wifi] connect %s: %s", ssid, result)
    _last_connect = {"status": "ok" if result.startswith("OK:") else "error",
                     "message": result}


@router.post("/connect", status_code=202)
def wifi_connect_endpoint(req: ConnectRequest, background_tasks: BackgroundTasks):
    global _last_connect
    _last_connect = {"status": "connecting", "message": f"Connecting to {req.ssid}…"}
    background_tasks.add_task(_do_connect, req.ssid, req.password)
    return {"status": "connecting", "ssid": req.ssid}


@router.get("/connect-result")
def wifi_connect_result() -> dict:
    return _last_connect
