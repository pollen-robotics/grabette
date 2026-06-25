"""Device-side relay client — talks to the Docker fleet Space over HTTP.

The device connects OUTBOUND to the Space (NAT-friendly), authenticating with
its locally-stored HF token, and short-polls for commands. Short-polling is
deliberately simple (no WebSocket reconnect/heartbeat edge cases) and its
steady request traffic keeps a free-tier Space awake.

Loop: register (also acts as heartbeat) → poll → execute → report results.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

import aiohttp

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
        poll_interval: float = 2.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token_provider = token_provider
        self.device_id = device_id
        self.name = name or device_id
        self.capabilities = capabilities or []
        self.poll_interval = poll_interval
        self.status = "offline"

    def _headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    async def _register(self, session: aiohttp.ClientSession, token: str) -> None:
        body = {"device_id": self.device_id, "name": self.name, "capabilities": self.capabilities}
        async with session.post(
            f"{self.base_url}/api/devices/register", json=body, headers=self._headers(token)
        ) as r:
            r.raise_for_status()

    async def _poll(self, session: aiohttp.ClientSession, token: str) -> list[dict[str, Any]]:
        async with session.get(
            f"{self.base_url}/api/devices/poll",
            params={"device_id": self.device_id},
            headers=self._headers(token),
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
        """Register + poll + dispatch + report, forever. Resilient to errors."""
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            registered = False
            while True:
                token = self.token_provider()
                if not token:
                    self.status, registered = "no-token", False
                    await asyncio.sleep(self.poll_interval)
                    continue
                try:
                    if not registered:
                        await self._register(session, token)
                        registered = True
                    commands = await self._poll(session, token)
                    for cmd in commands:
                        try:
                            res = await handler(cmd)
                        except Exception as e:  # noqa: BLE001
                            res = {"status": "error", "message": str(e)}
                        await self._report(session, token, cmd["id"], res)
                    self.status = "online"
                except aiohttp.ClientResponseError as e:
                    # 401/403 (token) or 404 (state lost on Space restart) → re-register
                    self.status, registered = f"http {e.status}", False
                    logger.warning("relay error %s; will re-register", e.status)
                    await asyncio.sleep(self.poll_interval)
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    self.status, registered = "unreachable", False
                    logger.debug("relay unreachable: %s", e)
                    await asyncio.sleep(self.poll_interval * 2)
                await asyncio.sleep(self.poll_interval)
