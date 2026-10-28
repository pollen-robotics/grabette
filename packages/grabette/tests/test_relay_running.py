"""The `running` list on the heartbeat and the result retry, against the real
RelayClient.run loop with the network faked out."""
import asyncio

import grabette.relay_client as rc


class _Resp:
    def __init__(self, payload=None, fail=False):
        self._payload = payload or {}
        self._fail = fail

    def raise_for_status(self):
        if self._fail:
            raise RuntimeError("HTTP 502")

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    """Hands out `commands` once, records heartbeats, fails the first
    `failing_reports` results."""

    def __init__(self, commands, failing_reports=0):
        self._commands = list(commands)
        self.beats: list[dict] = []
        self.reports: list[str] = []
        self.report_calls = 0
        self._failing = failing_reports

    def get(self, url, **kw):  # /poll
        cmds, self._commands = self._commands, []
        return _Resp({"commands": cmds})

    def post(self, url, **kw):
        if url.endswith("/heartbeat"):
            self.beats.append(dict(kw.get("params") or {}))
            return _Resp()
        if url.endswith("/result"):
            self.report_calls += 1
            if self.report_calls <= self._failing:
                return _Resp(fail=True)
            self.reports.append(kw["json"]["command_id"])
            return _Resp()
        return _Resp()  # /register

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _client(monkeypatch, session):
    monkeypatch.setattr(rc.aiohttp, "ClientSession", lambda *a, **kw: session)
    monkeypatch.setattr(rc, "get_route_ip", lambda: "10.0.0.2")
    monkeypatch.setattr(rc, "_HEARTBEAT_INTERVAL_S", 0.01)
    monkeypatch.setattr(rc, "_REPORT_RETRY_S", 0.01)

    async def no_wifi():
        return ""

    client = rc.RelayClient("https://fleet.test", lambda: "tok", "dev-1",
                            poll_interval=0.01)
    monkeypatch.setattr(client, "_wifi_network", no_wifi)
    return client


async def _until(cond, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        assert loop.time() < end, "condition never met"
        await asyncio.sleep(0.01)


def test_a_held_command_is_listed_on_every_beat_until_it_is_reported(monkeypatch):
    session = _Session([{"id": "up1", "type": "upload_episodes", "args": {}}])
    client = _client(monkeypatch, session)
    release = asyncio.Event()

    async def handler(cmd):
        await release.wait()
        return {"status": "ok"}

    async def go():
        task = asyncio.create_task(client.run(handler))
        try:
            await _until(lambda: sum(b.get("running") == "up1" for b in session.beats) >= 3)
            release.set()
            await _until(lambda: session.reports == ["up1"])
            n = len(session.beats)
            await _until(lambda: len(session.beats) >= n + 2)
        finally:
            task.cancel()

    asyncio.run(go())

    assert all("running" in b for b in session.beats)
    assert session.beats[-1]["running"] == ""


def test_a_failed_result_is_retried_and_stays_listed_meanwhile(monkeypatch):
    session = _Session([{"id": "up1", "type": "upload_episodes", "args": {}}],
                       failing_reports=2)
    client = _client(monkeypatch, session)

    async def handler(cmd):
        return {"status": "ok"}

    async def go():
        task = asyncio.create_task(client.run(handler))
        try:
            await _until(lambda: session.reports == ["up1"])
        finally:
            task.cancel()

    asyncio.run(go())

    assert session.report_calls == 3


def test_upload_progress_rides_on_the_heartbeat_only_while_there_is_some(monkeypatch):
    session = _Session([])
    client = _client(monkeypatch, session)
    progress = {"value": "up1:3/10"}
    client.progress_provider = lambda: progress["value"]

    async def handler(cmd):
        return {"status": "ok"}

    async def go():
        task = asyncio.create_task(client.run(handler))
        try:
            await _until(lambda: any(b.get("progress") == "up1:3/10" for b in session.beats))
            progress["value"] = ""
            n = len(session.beats)
            await _until(lambda: len(session.beats) >= n + 2)
        finally:
            task.cancel()

    asyncio.run(go())

    # Absent rather than empty once the upload is over: an older fleet ignores
    # the field either way, and a newer one reads absence as "no upload".
    assert "progress" not in session.beats[-1]
