"""Device-side relay client — talks to the Docker fleet Space over HTTP.

The device connects OUTBOUND to the Space (NAT-friendly), authenticating with
its locally-stored HF token, and polls for commands. NAT-friendly outbound
polling avoids WebSocket reconnect/heartbeat edge cases, and the request
traffic keeps a free-tier Space awake.

The transport auto-adapts to the server: if the fleet long-polls (holds the
GET open until a command is queued), delivery is a network round-trip and the
client re-polls immediately; if the fleet short-polls (answers instantly), the
client throttles to poll_interval. No client config needed — the server's
LONG_POLL_S alone decides, so it can be flipped without touching devices.

Loop: register (also acts as heartbeat) → poll → execute → report results.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

import aiohttp

from grabette.wifi import get_route_ip

# Per-request timeout for the poll GET. Must exceed the fleet's LONG_POLL_S hold
# (server-side, ~25s) so a legitimately held poll isn't cut short and mistaken
# for a network failure. Register/result keep the shorter session default.
_POLL_TIMEOUT_S = 60.0
# If a poll returned no commands but took at least this long, the server held it
# open (long-poll) → re-poll immediately. A near-instant empty response means
# the server is short-polling → throttle to poll_interval. Above normal
# short-poll latency (tens of ms), well below LONG_POLL_S.
_LONGPOLL_HOLD_HINT_S = 1.5

logger = logging.getLogger("grabette.relay_client")

CommandHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
TokenProvider = Callable[[], Optional[str]]


class RelayClient:
    def __init__(
        self,
        base_url: str,
        token_provider: TokenProvider,
        device_id: str,
        *,
        name: Optional[str] = None,
        capabilities: Optional[list[str]] = None,
        hand: Optional[str] = None,
        # Short poll interval so a peer receives a fanned-out command (notably a
        # group STOP) within ~1s of the acting device, keeping the group's stop
        # spread small without any scheduled lead. This is the short-polling
        # ceiling on delivery latency; long-polling (planned) would cut it to a
        # network round-trip. Trade-off: more HTTP requests to the fleet Space
        # (cheap, and it keeps a free-tier Space awake).
        poll_interval: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token_provider = token_provider
        self.device_id = device_id
        self.name = name or device_id
        self.capabilities = capabilities or []
        self.hand = hand or ""
        self.poll_interval = poll_interval
        self.status = "offline"

    def _headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    async def _register(self, session: aiohttp.ClientSession, token: str) -> None:
        body = {
            "device_id": self.device_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "hand": self.hand,
            "ip": get_route_ip(),  # recomputed each register so IP changes are caught
        }
        async with session.post(
            f"{self.base_url}/api/devices/register", json=body, headers=self._headers(token)
        ) as r:
            r.raise_for_status()

    async def _poll(self, session: aiohttp.ClientSession, token: str) -> list[dict[str, Any]]:
        async with session.get(
            f"{self.base_url}/api/devices/poll",
            params={"device_id": self.device_id},
            headers=self._headers(token),
            timeout=aiohttp.ClientTimeout(total=_POLL_TIMEOUT_S),
        ) as r:
            r.raise_for_status()
            return (await r.json()).get("commands", [])

    async def _report(
        self, session: aiohttp.ClientSession, token: str, command_id: str, result: dict[str, Any]
    ) -> None:
        body = {"device_id": self.device_id, "command_id": command_id, "result": result}
        async with session.post(
            f"{self.base_url}/api/devices/result", json=body, headers=self._headers(token)
        ) as r:
            r.raise_for_status()

    async def run(self, handler: CommandHandler) -> None:
        """Register + poll + dispatch + report, forever. Resilient to errors.

        The poll loop and command EXECUTION are decoupled: the loop only
        registers, polls, and enqueues commands, then keeps polling. A separate
        worker executes commands and reports results. This matters because some
        handlers block for seconds (stop_capture muxes the mp4 + tears down the
        OAK-D). If the poll loop awaited them inline, the device would stop
        heartbeating AND stop receiving commands for the whole muxing window —
        so it'd flap offline and the NEXT episode's start_capture would arrive
        late (past its T0). Decoupled, the loop keeps the device online and
        delivers commands promptly; the worker runs them (in order, one at a
        time — the backend can't record two captures at once) as it frees up."""
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            queue: "asyncio.Queue[dict]" = asyncio.Queue()
            inflight: set[str] = set()  # command ids queued/running (dedup)

            async def worker() -> None:
                while True:
                    cmd = await queue.get()
                    try:
                        try:
                            res = await handler(cmd)
                        except Exception as e:  # noqa: BLE001
                            res = {"status": "error", "message": str(e)}
                        token = self.token_provider()
                        if token:
                            try:
                                await self._report(session, token, cmd["id"], res)
                            except Exception:
                                logger.warning("relay report failed for %s", cmd.get("id"), exc_info=True)
                    finally:
                        inflight.discard(cmd.get("id"))
                        queue.task_done()

            worker_task = asyncio.create_task(worker())
            registered = False
            try:
                while True:
                    token = self.token_provider()
                    if not token:
                        self.status, registered = "no-token", False
                        await asyncio.sleep(self.poll_interval)
                        continue
                    # Delay before the NEXT poll. Default = throttle by
                    # poll_interval (short-poll server, or after an error). Set
                    # to 0 to re-poll immediately when the server long-polled
                    # (held the connection) or handed us work — no throttling
                    # needed, the hold itself paced us.
                    delay = self.poll_interval
                    try:
                        if not registered:
                            await self._register(session, token)
                            registered = True
                        t0 = time.monotonic()
                        commands = await self._poll(session, token)
                        elapsed = time.monotonic() - t0
                        for cmd in commands:
                            cid = cmd.get("id")
                            if cid in inflight:
                                continue  # already queued/running — don't double-dispatch
                            inflight.add(cid)
                            queue.put_nowait(cmd)
                        self.status = "online"
                        # Auto-detect the server's mode: work returned, or the
                        # server held the poll open (long-poll) → re-poll now.
                        # An instant empty response means short-poll → throttle.
                        if commands or elapsed >= _LONGPOLL_HOLD_HINT_S:
                            delay = 0.0
                    except aiohttp.ClientResponseError as e:
                        # 401/403 (token) or 404 (state lost on Space restart) → re-register
                        self.status, registered = f"http {e.status}", False
                        logger.warning("relay error %s; will re-register", e.status)
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        self.status, registered = "unreachable", False
                        logger.debug("relay unreachable: %s", e)
                        delay = self.poll_interval * 2
                    except Exception:
                        # Anything unexpected must NOT kill the loop — the relay
                        # is meant to run forever. Log, re-register, keep going.
                        self.status, registered = "error", False
                        logger.exception("relay loop error; continuing")
                    if delay:
                        await asyncio.sleep(delay)
            finally:
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
