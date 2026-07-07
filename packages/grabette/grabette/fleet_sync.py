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


async def request_group_start(task_name: str | None) -> Optional[dict[str, Any]]:
    """Returns the fleet response dict on success, or None to start solo.

    Response is {"status": "solo"} when the device has no group (or is the
    only member), or {"status": "scheduled", "scheduled_start_utc": ...,
    "task_name": ..., "peers": [...]} when peers were notified.
    """
    headers = _auth_headers()
    if headers is None:
        return None
    import aiohttp

    url = f"{settings.relay_url.rstrip('/')}/api/devices/{settings.device_id}/sync/start"
    try:
        timeout = aiohttp.ClientTimeout(total=_START_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json={"task_name": task_name or ""}, headers=headers) as r:
                if r.status != 200:
                    logger.warning("fleet sync/start failed: HTTP %d — %s", r.status, await r.text())
                    return None
                return await r.json()
    except Exception as e:  # noqa: BLE001 — any network failure degrades to solo start
        logger.info("fleet sync/start unavailable (%s) — starting solo", e)
        return None


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
