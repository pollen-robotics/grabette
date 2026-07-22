"""Ask the fleet broker to orchestrate a synchronized group start/stop.

Called by every local start/stop trigger (physical button, local UI) so a
device grouped in grabette-fleet stays in lockstep with its group peers no
matter which physical device the user actually acted on. Best-effort: when
the device isn't grouped, isn't logged in, or the fleet Space is unreachable,
these return None / do nothing and the caller falls back to a plain local
start/stop — unchanged from before group sync existed.

Timeouts are kept short (a few seconds) so a slow or sleeping fleet Space
degrades to "start solo" rather than stalling every local recording.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from grabette.config import settings

logger = logging.getLogger(__name__)

_START_TIMEOUT_S = 3.0
_STOP_TIMEOUT_S = 3.0


def _auth_headers() -> Optional[dict[str, str]]:
    if not settings.relay_enabled or not settings.device_id:
        return None
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


async def request_group_start(task_name: str | None) -> dict[str, Any]:
    """Ask the fleet to orchestrate a group start. Always returns a dict with
    an explicit "status" so the caller never confuses these cases:

      * "solo"       — fleet says this device is in no open session; record
                       locally (legitimate standalone episode).
      * "scheduled"  — group start scheduled: {scheduled_start_utc, task_name,
                       peers}. Record at the shared T0.
      * "refused"    — fleet was REACHED but declined (e.g. a peer is offline,
                       HTTP 409). The device IS in a group session, so the
                       caller must NOT silently record a half-rig solo episode
                       — it should abort. Carries {http, detail}.
      * "unreachable"— no token / relay disabled / network error. Fleet's view
                       is unknown, so fall back to a local solo recording
                       (device stays useful standalone).
    """
    headers = _auth_headers()
    if headers is None:
        return {"status": "unreachable"}
    import aiohttp

    url = f"{settings.relay_url.rstrip('/')}/api/devices/{settings.device_id}/sync/start"
    try:
        timeout = aiohttp.ClientTimeout(total=_START_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json={"task_name": task_name or ""}, headers=headers) as r:
                if r.status == 200:
                    return await r.json()
                detail = await r.text()
                logger.warning("fleet sync/start refused: HTTP %d — %s", r.status, detail)
                return {"status": "refused", "http": r.status, "detail": detail}
    except Exception as e:  # noqa: BLE001 — network failure → standalone solo
        logger.info("fleet sync/start unreachable (%s) — recording solo", e)
        return {"status": "unreachable"}


async def notify_group_stop() -> None:
    """Best-effort, fire-and-forget: tell fleet to fan a stop out to peers.

    Never raises — a failure here must not affect the local stop that
    already happened.
    """
    headers = _auth_headers()
    if headers is None:
        return
    import aiohttp

    url = f"{settings.relay_url.rstrip('/')}/api/devices/{settings.device_id}/sync/stop"
    try:
        timeout = aiohttp.ClientTimeout(total=_STOP_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers) as r:
                if r.status != 200:
                    logger.warning("fleet sync/stop failed: HTTP %d — %s", r.status, await r.text())
    except Exception as e:  # noqa: BLE001
        logger.info("fleet sync/stop unavailable (%s)", e)
