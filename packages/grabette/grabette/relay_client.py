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

from grabette.wifi import get_current_ssid, get_route_ip

# Per-request timeout for the poll GET. Must exceed the fleet's LONG_POLL_S hold
# (server-side, ~25s) so a legitimately held poll isn't cut short and mistaken
# for a network failure. Register/result keep the shorter session default.
_POLL_TIMEOUT_S = 60.0
# If a poll returned no commands but took at least this long, the server held it
# open (long-poll) → re-poll immediately. A near-instant empty response means
# the server is short-polling → throttle to poll_interval. Above normal
# short-poll latency (tens of ms), well below LONG_POLL_S.
_LONGPOLL_HOLD_HINT_S = 1.5
# Lightweight liveness ping cadence, INDEPENDENT of the (long-held) command poll:
# it keeps the fleet's last_seen fresh so a disconnect is detected within the
# fleet's ONLINE_WINDOW instead of waiting out the long-poll hold. Keep in step
# with the fleet's DEVICE_HEARTBEAT_S.
_HEARTBEAT_INTERVAL_S = 5.0
# Battery is read on its OWN slow cadence, NOT inside the heartbeat: the read is
# I2C (PiSugar) and I2C is contended during capture, so a read can stall for
# seconds. Doing it in the heartbeat would stall liveness → the fleet would mark
# the device offline mid-recording and drop its queued start/stop commands. So
# we cache the last value here and the heartbeat just ships the cache (instant).
_BATTERY_INTERVAL_S = 30.0
# Commands that must NOT go through the serialized worker below. The worker runs
# one command at a time, so a cancel queued behind the upload it is cancelling
# would only run once that upload had finished — i.e. never cancel anything. These
# are dispatched immediately, concurrently with whatever is running. Only put
# command types here that do no hardware work and return fast.
_FAST_PATH_TYPES = frozenset({"cancel_dataset"})

# Hardware-fault text is reported on the heartbeat, i.e. in a query string. The
# messages name the fault, the consequence and the fix, so they are a sentence or
# two — long enough to be worth capping, short enough that the cap never bites.
_FAULT_MAX_CHARS = 300

logger = logging.getLogger("grabette.relay_client")

CommandHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
TokenProvider = Callable[[], Optional[str]]

# The relay's live HTTP session, published for the other device-side callers that
# have to reach the fleet (fleet_sync's group start/stop). Borrowing it means those
# calls ride an already-open keep-alive connection instead of paying DNS + TCP +
# TLS on a brand-new ClientSession — which, on a Pi whose CPU and WiFi are both
# saturated by an in-flight capture, is what pushed a group stop past its timeout
# and left a peer recording. The held long-poll does NOT block them: aiohttp's
# connector allows unlimited connections per host, so a stop opens its own rather
# than queueing behind the poll.
_live_session: Optional[aiohttp.ClientSession] = None


def get_relay_session() -> Optional[aiohttp.ClientSession]:
    """The relay's live session, or None when the relay isn't running (callers
    then open their own). Borrowed — never close the returned session."""
    s = _live_session
    return s if (s is not None and not s.closed) else None


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
        # Optional callable returning the battery percentage (0-100) or None.
        # Piggy-backed on the heartbeat so the fleet can show each device's
        # charge without polling. Called off the event loop (may do I2C).
        battery_provider: Optional[Callable[[], Optional[float]]] = None,
        # Optional callable returning this device's recorded tasks (see
        # TaskManager.report_tasks). Sent on register so the fleet — regardless
        # of which HF account it runs under — can surface the device's tasks and
        # name each episode's peers. The device is the durable source of truth.
        tasks_provider: Optional[Callable[[], list[dict]]] = None,
        # Optional callable returning this device's loose episodes — recorded
        # outside any task (see TaskManager.report_unassigned). Sent on register
        # beside `tasks` but on its OWN key: they belong to no task, and must not
        # be merged across devices the way tasks are. No extra freshness wiring
        # needed — filing an episode bumps the task revision below, which already
        # triggers a re-report on the next beat.
        unassigned_provider: Optional[Callable[[], dict]] = None,
        # Optional callable returning a monotonic revision that changes whenever
        # this device's tasks change (see TaskManager.revision). Watched on the
        # heartbeat so a freshly-recorded episode is re-reported to the fleet
        # within one beat, instead of only on the next reconnect.
        tasks_rev_provider: Optional[Callable[[], int]] = None,
        # Optional callable returning this device's current activity — one of
        # "idle" | "capturing" | "uploading" | "processing" — piggy-backed on the
        # heartbeat so the operator dashboard can show device state and block a
        # recording while the device is busy. In-memory & fast (unlike battery's
        # I2C), so it's read inline on the heartbeat path.
        activity_provider: Optional[Callable[[], Optional[str]]] = None,
        # Optional callable returning why this device is in a HARDWARE FAULT —
        # a state where it refuses to record because every episode would be
        # unconvertible (no OAK-D calibration, no angle sensors) — or "" when
        # healthy. A separate channel from activity on purpose: the two are
        # orthogonal (a faulted device can also be uploading), and folding a
        # fault into the activity enum would have made it invisible to a fleet
        # that only knows the four activity values.
        fault_provider: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token_provider = token_provider
        self.device_id = device_id
        self.name = name or device_id
        self.capabilities = capabilities or []
        self.hand = hand or ""
        self.poll_interval = poll_interval
        self.battery_provider = battery_provider
        self.tasks_provider = tasks_provider
        self.unassigned_provider = unassigned_provider
        self.tasks_rev_provider = tasks_rev_provider
        self.activity_provider = activity_provider
        self.fault_provider = fault_provider
        self._last_reported_rev: Optional[int] = None  # last task revision sent to the fleet
        self._battery: Optional[float] = None  # cached; refreshed off the heartbeat path
        self.status = "offline"

    def _headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    async def _wifi_network() -> str:
        """Current WiFi SSID, or "" if unknown (wired, hotspot, no nmcli).

        Reported alongside the IP so the fleet page can say WHICH network a device
        is on — two devices on different LANs both look plainly "online" otherwise.
        Off the event loop: get_current_ssid() shells out to nmcli, and this runs
        on the register path (which doubles as the heartbeat)."""
        try:
            return await asyncio.to_thread(get_current_ssid) or ""
        except Exception:
            logger.debug("could not read the WiFi SSID", exc_info=True)
            return ""

    async def _register(self, session: aiohttp.ClientSession, token: str) -> None:
        body = {
            "device_id": self.device_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "hand": self.hand,
            "ip": get_route_ip(),  # recomputed each register so IP changes are caught
            # Recomputed too — a device moved to another WiFi keeps the same id.
            "network": await self._wifi_network(),
        }
        if self.tasks_provider is not None:
            try:
                body["tasks"] = self.tasks_provider()
            except Exception:
                logger.debug("tasks_provider failed", exc_info=True)
        if self.unassigned_provider is not None:
            try:
                body["unassigned"] = self.unassigned_provider()
            except Exception:
                logger.debug("unassigned_provider failed", exc_info=True)
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
        time — the backend can't record two captures at once) as it frees up.

        One exception to that serialization: _FAST_PATH_TYPES (cancels) are run
        immediately, beside the worker. A cancel queued behind the multi-minute
        upload it is cancelling would be pointless."""
        global _live_session
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Publish it for fleet_sync (see get_relay_session), so a group stop
            # doesn't pay a cold TLS handshake at the worst possible moment.
            _live_session = session
            queue: "asyncio.Queue[dict]" = asyncio.Queue()
            inflight: set[str] = set()  # command ids queued/running (dedup)
            # Strong refs to fast-path tasks: asyncio only weakly references tasks,
            # so without this the GC can cancel a cancel mid-flight.
            fast_tasks: set[asyncio.Task] = set()

            async def run_one(cmd: dict) -> None:
                """Execute one command and report its result."""
                from grabette.cancel import get_cancel_registry
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
                    # Nothing can act on this command any more, so stop remembering
                    # a cancel for it.
                    get_cancel_registry().clear(cmd.get("id"))
                    inflight.discard(cmd.get("id"))

            async def worker() -> None:
                while True:
                    cmd = await queue.get()
                    try:
                        await run_one(cmd)
                    finally:
                        queue.task_done()

            async def heartbeat_loop() -> None:
                """Ping /api/devices/heartbeat every few seconds, independently of
                the held command poll, so the fleet sees us as alive and detects a
                real disconnect fast. Best-effort: a missed beat only delays
                detection slightly; a 404 (not yet registered) is ignored."""
                while True:
                    await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
                    token = self.token_provider()
                    if not token:
                        continue
                    params = {"device_id": self.device_id}
                    if self._battery is not None:  # cached value only — never I2C here
                        params["battery"] = self._battery
                    if self.activity_provider is not None:  # in-memory, safe to read inline
                        try:
                            act = self.activity_provider()
                        except Exception:
                            act = None
                        if act:
                            params["status"] = act
                    if self.fault_provider is not None:  # in-memory, safe inline
                        try:
                            fault = self.fault_provider()
                        except Exception:
                            fault = None
                        # Sent on EVERY beat, empty included: the fleet must learn
                        # that a fault cleared as promptly as it learned of it, and
                        # a field that only appears when broken can never say
                        # "fixed". Truncated because it rides in a query string.
                        params["error"] = (fault or "")[:_FAULT_MAX_CHARS]
                    try:
                        async with session.post(
                            f"{self.base_url}/api/devices/heartbeat",
                            params=params,
                            headers=self._headers(token),
                            timeout=aiohttp.ClientTimeout(total=10),
                        ):
                            pass
                    except Exception:
                        pass
                    # Re-report tasks when they've changed since the last report
                    # (e.g. a just-recorded episode), so the fleet's aggregated
                    # view stays fresh without resending the full list every beat.
                    if self.tasks_rev_provider is not None:
                        try:
                            rev = self.tasks_rev_provider()
                        except Exception:
                            rev = self._last_reported_rev
                        if rev != self._last_reported_rev:
                            try:
                                await self._register(session, token)
                                self._last_reported_rev = rev
                            except Exception:
                                pass  # retry on the next beat

            async def battery_loop() -> None:
                """Refresh the cached battery % on a slow cadence, isolated from
                the heartbeat: the I2C read can stall under capture contention,
                but that only makes the reading stale — it never delays liveness."""
                if self.battery_provider is None:
                    return
                while True:
                    try:
                        self._battery = await asyncio.to_thread(self.battery_provider)
                    except Exception:
                        pass
                    await asyncio.sleep(_BATTERY_INTERVAL_S)

            worker_task = asyncio.create_task(worker())
            heartbeat_task = asyncio.create_task(heartbeat_loop())
            battery_task = asyncio.create_task(battery_loop())
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
                            # Mark the just-sent task revision so the heartbeat
                            # only re-reports on subsequent changes.
                            if self.tasks_rev_provider is not None:
                                try:
                                    self._last_reported_rev = self.tasks_rev_provider()
                                except Exception:
                                    pass
                        t0 = time.monotonic()
                        commands = await self._poll(session, token)
                        elapsed = time.monotonic() - t0
                        for cmd in commands:
                            cid = cmd.get("id")
                            if cid in inflight:
                                continue  # already queued/running — don't double-dispatch
                            inflight.add(cid)
                            if cmd.get("type") in _FAST_PATH_TYPES:
                                # Run it NOW, beside the worker: a cancel must not
                                # wait for the long job it cancels (_FAST_PATH_TYPES).
                                fast = asyncio.create_task(run_one(cmd))
                                fast_tasks.add(fast)
                                fast.add_done_callback(fast_tasks.discard)
                            else:
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
                # Stop handing out a session that's about to be closed.
                _live_session = None
                worker_task.cancel()
                heartbeat_task.cancel()
                battery_task.cancel()
                for t in list(fast_tasks):
                    t.cancel()
                for t in (worker_task, heartbeat_task, battery_task):
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
