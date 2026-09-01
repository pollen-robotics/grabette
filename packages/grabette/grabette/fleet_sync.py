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

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from grabette.config import settings

logger = logging.getLogger(__name__)

_START_TIMEOUT_S = 3.0
# First stop attempt. Kept short because it BLOCKS the button press: the operator
# must never wait on the fleet before their own device stops.
_STOP_TIMEOUT_S = 3.0
# Background retries, once the local stop is no longer waiting on us. Looser
# per-attempt budget, but few and quick on purpose — see _retry_group_stop.
_STOP_RETRY_TIMEOUT_S = 4.0
_STOP_RETRIES = 2
_STOP_RETRY_DELAY_S = 1.0

# Strong references to in-flight retry tasks. asyncio holds tasks only weakly, so
# without this the GC can cancel a retry mid-flight — leaving the peers recording,
# which is the exact failure the retry exists to prevent.
_retry_tasks: set[asyncio.Task] = set()

# _post_group_stop outcomes.
_OK = "ok"              # the fleet took it (or there was nothing to send)
_RETRY = "retry"        # transient (network, timeout, 5xx) — worth another go
_GIVE_UP = "give-up"    # 4xx: our request is wrong, retrying can't fix it


def _auth_headers() -> Optional[dict[str, str]]:
    if not settings.relay_enabled or not settings.device_id:
        return None
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


@asynccontextmanager
async def _http_session():
    """The relay's live session when it's up, else a private one.

    Borrowing matters most on the stop path: a fresh ClientSession pays DNS + TCP
    + TLS to the Space while the Pi is flat out recording, and that setup cost
    alone can eat the whole timeout. Falls back to a private session so a device
    with the relay disabled keeps working exactly as before."""
    from grabette.relay_client import get_relay_session

    shared = get_relay_session()
    if shared is not None:
        yield shared  # borrowed — must NOT be closed here
        return
    import aiohttp

    async with aiohttp.ClientSession() as own:
        yield own


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
        # Timeout passed per-request: the borrowed relay session carries its own
        # (longer) session-wide default, which must not silently apply here.
        async with _http_session() as session:
            async with session.post(url, json={"task_name": task_name or ""},
                                    headers=headers, timeout=timeout) as r:
                if r.status == 200:
                    return await r.json()
                detail = await r.text()
                logger.warning("fleet sync/start refused: HTTP %d — %s", r.status, detail)
                return {"status": "refused", "http": r.status, "detail": detail}
    except Exception as e:  # noqa: BLE001 — network failure → standalone solo
        logger.info("fleet sync/start unreachable (%s) — recording solo", e)
        return {"status": "unreachable"}


async def _post_group_stop(timeout_s: float) -> str:
    """One POST to the fleet's group-stop endpoint. Returns _OK / _RETRY / _GIVE_UP.

    Re-sending is safe: the fleet re-arms the peer's already-queued stop_capture
    instead of duplicating it, and a stop_capture on a device that has already
    stopped is a no-op."""
    headers = _auth_headers()
    if headers is None:
        return _OK  # relay off / not logged in — no peers to notify, not a failure
    import aiohttp

    url = f"{settings.relay_url.rstrip('/')}/api/devices/{settings.device_id}/sync/stop"
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with _http_session() as session:
            async with session.post(url, headers=headers, timeout=timeout) as r:
                if r.status == 200:
                    return _OK
                detail = await r.text()
                if 400 <= r.status < 500:
                    # Unknown device, bad token… our own request is wrong; another
                    # attempt would fail identically.
                    logger.error("fleet sync/stop rejected: HTTP %d — %s", r.status, detail)
                    return _GIVE_UP
                logger.warning("fleet sync/stop failed: HTTP %d — %s", r.status, detail)
                return _RETRY
    except Exception as e:  # noqa: BLE001
        logger.warning("fleet sync/stop unreachable (%s)", e)
        return _RETRY


async def notify_group_stop(should_abort: Optional[Callable[[], bool]] = None) -> None:
    """Tell the fleet to fan a stop out to the group's peers.

    Called BEFORE the local stop_capture (which blocks the loop for the whole mux),
    so the peers stop within ~1 poll interval of us instead of trailing our teardown.
    That's why the first attempt is short-budgeted: it delays the press.

    A single lost call used to leave the peers recording indefinitely and the fleet
    session stuck on "recording" — nothing anywhere retried or reconciled it. So a
    transient failure now hands off to background retries and returns at once,
    keeping the press responsive.

    should_abort: consulted before each retry; True cancels the remaining ones.
    Callers pass "a capture is running again", so a retry can never land on a NEW
    episode and stop it. Never raises — a failure here must not disturb the local
    stop, which is the one thing that must always happen."""
    if await _post_group_stop(_STOP_TIMEOUT_S) != _RETRY:
        return
    task = asyncio.create_task(_retry_group_stop(should_abort))
    _retry_tasks.add(task)
    task.add_done_callback(_retry_tasks.discard)


async def _retry_group_stop(should_abort: Optional[Callable[[], bool]]) -> None:
    """Retry the stop fan-out in the background, after the local stop.

    The whole window is kept short (~10s worst case) on purpose: a retry landing
    after the operator has launched the NEXT episode would stop that one instead.
    should_abort closes that hole outright; the tight budget is the belt to its
    braces. Anything still lost after this is caught by the fleet's own watchdog,
    which notices a member reporting idle while its session believes it's recording."""
    for attempt in range(1, _STOP_RETRIES + 1):
        await asyncio.sleep(_STOP_RETRY_DELAY_S)
        try:
            if should_abort is not None and should_abort():
                logger.warning("fleet sync/stop: capture active again — abandoning the retry")
                return
            outcome = await _post_group_stop(_STOP_RETRY_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — a background task must never die noisily
            logger.exception("fleet sync/stop retry %d crashed", attempt)
            continue
        if outcome == _OK:
            logger.warning("fleet sync/stop went through on retry %d", attempt)
            return
        if outcome == _GIVE_UP:
            return
    logger.error(
        "fleet sync/stop failed after %d attempts — the group's peers were NOT told to stop "
        "and may still be recording. The fleet's watchdog should catch this shortly.",
        _STOP_RETRIES + 1,
    )
