"""Fleet command-dispatcher tests (_handle_relay_command).

Exercises the recording subset casquette handles as a pure fleet peer:
get_state, cancel_dataset, immediate + synchronized (T0) start/stop, and the
too-late-start guard — all against the mock backend + a tmp task store.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from casquette.app.main import _handle_relay_command


def _utc_iso(offset_s: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


async def test_get_state_reports_running(daemon, tm):
    r = await _handle_relay_command({"type": "get_state", "id": "c1"})
    assert r["status"] == "ok"
    assert r["state"]["state"] == "running"


async def test_cancel_dataset_marks_ids(daemon, tm):
    r = await _handle_relay_command(
        {"type": "cancel_dataset", "id": "c2", "args": {"command_ids": ["j1"]}}
    )
    assert r["status"] == "ok"
    assert r["cancelled"] == ["j1"]


async def test_immediate_start_stop_records_episode(daemon, tm):
    r = await _handle_relay_command(
        {"type": "start_capture", "id": "c3", "args": {"task_name": "pick"}}
    )
    assert r["status"] == "ok"
    episode_id = r["episode_id"]
    assert daemon.backend.is_capturing

    r2 = await _handle_relay_command({"type": "stop_capture", "id": "c4", "args": {}})
    assert r2["status"] == "ok"
    assert not daemon.backend.is_capturing

    # The episode is filed under the "pick" task in the record store.
    tasks = {t["name"]: t for t in tm.report_tasks()}
    assert "pick" in tasks
    assert episode_id in tm.episode_dir(episode_id).name
    assert tm.episode_dir(episode_id).exists()


async def test_start_rejected_when_already_capturing(daemon, tm):
    await _handle_relay_command(
        {"type": "start_capture", "id": "c5", "args": {"task_name": "pick"}}
    )
    r = await _handle_relay_command(
        {"type": "start_capture", "id": "c6", "args": {"task_name": "pick"}}
    )
    assert r["status"] == "error"
    await _handle_relay_command({"type": "stop_capture", "id": "c7", "args": {}})


async def test_scheduled_start_shares_t0_episode_id_and_fires(daemon, tm):
    t0 = _utc_iso(0.3)
    r = await _handle_relay_command(
        {"type": "start_capture", "id": "c8",
         "args": {"task_name": "pick", "start_at_utc": t0}}
    )
    assert r["status"] == "scheduled"
    # episode id is derived from the shared T0 (same on every device), not local
    # wall-clock creation time.
    assert r["episode_id"] == datetime.fromisoformat(t0).astimezone(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    await asyncio.sleep(0.6)  # wait past T0
    assert daemon.backend.is_capturing
    r2 = await _handle_relay_command({"type": "stop_capture", "id": "c9", "args": {}})
    assert r2["status"] == "ok"


async def test_start_too_late_is_refused(daemon, tm):
    r = await _handle_relay_command(
        {"type": "start_capture", "id": "c10",
         "args": {"task_name": "pick", "start_at_utc": _utc_iso(-5)}}
    )
    assert r["status"] == "error"
    assert "late" in r["message"]
    assert not daemon.backend.is_capturing


async def test_unknown_command(daemon, tm):
    r = await _handle_relay_command({"type": "frobnicate", "id": "c11"})
    assert r["status"] == "error"
    assert "unknown" in r["message"]


async def test_upload_episodes_requires_repo_and_role(daemon, tm):
    r = await _handle_relay_command(
        {"type": "upload_episodes", "id": "u1", "args": {"episode_ids": ["x"]}}
    )
    assert r["status"] == "error"
    assert "raw_repo" in r["message"]


class _FakeHF:
    def __init__(self, on_hub=(), fail_at=None, error=None):
        self.commits, self.messages = [], []
        self._on_hub, self._fail_at, self._error = on_hub, fail_at, error

    def already_uploaded(self, episodes, repo):
        if isinstance(self._on_hub, Exception):
            raise self._on_hub
        return {dest for _, dest in episodes if dest in self._on_hub}

    def upload_episodes(self, batch, repo, private, message=None):
        self.commits.append([dest for _, dest in batch])
        self.messages.append(message)
        if self._fail_at == len(self.commits):
            raise self._error


async def _upload(tm, monkeypatch, hf, episode_ids):
    import casquette.fleet.hf as hfmod

    monkeypatch.setattr(hfmod, "get_hf_client", lambda: hf)
    return await _handle_relay_command({
        "type": "upload_episodes", "id": "u2",
        "args": {"raw_repo": "org/ds", "role": "casquette", "episode_ids": episode_ids},
    })


async def test_upload_episodes_uploads_present_flags_missing(daemon, tm, monkeypatch):
    eid = tm.create_episode(tm.get_or_create_task("pick"))  # creates the dir
    hf = _FakeHF()

    r = await _upload(tm, monkeypatch, hf, [eid, "20990101_000000"])

    assert r["status"] == "ok"
    assert r["uploaded"] == [eid]
    assert r["missing"] == ["20990101_000000"]
    # each stream uploaded under "{episode_id}/{role}"
    assert hf.commits == [[f"{eid}/casquette"]]


async def test_upload_episodes_share_a_commit_per_batch(daemon, tm, monkeypatch):
    # One commit per episode per device exhausted the Hub's per-repo commit
    # quota (320/hour) on a single build.
    import casquette.app.main as main

    monkeypatch.setattr(main, "_UPLOAD_BATCH_EPISODES", 2)
    task = tm.get_or_create_task("pick")
    eids = [tm.create_episode(task, episode_id=f"2026010{i}_000000") for i in range(3)]
    hf = _FakeHF()

    r = await _upload(tm, monkeypatch, hf, eids)

    assert r["status"] == "ok" and r["uploaded"] == eids
    assert hf.commits == [[f"{e}/casquette" for e in eids[:2]], [f"{eids[2]}/casquette"]]
    assert hf.messages == ["batch 1/2 (casquette)", "batch 2/2 (casquette)"]


async def test_upload_episodes_already_on_the_hub_are_skipped(daemon, tm, monkeypatch):
    task = tm.get_or_create_task("pick")
    eids = [tm.create_episode(task, episode_id=f"2026010{i}_000000") for i in range(2)]
    hf = _FakeHF(on_hub={f"{eids[0]}/casquette"})

    r = await _upload(tm, monkeypatch, hf, eids)

    assert r["status"] == "ok" and sorted(r["uploaded"]) == sorted(eids)
    assert hf.commits == [[f"{eids[1]}/casquette"]]


async def test_upload_episodes_failed_listing_uploads_everything(daemon, tm, monkeypatch):
    eid = tm.create_episode(tm.get_or_create_task("pick"))
    hf = _FakeHF(on_hub=ConnectionResetError("reset"))

    r = await _upload(tm, monkeypatch, hf, [eid])

    assert r["status"] == "ok" and hf.commits == [[f"{eid}/casquette"]]


async def test_upload_episodes_rate_limit_is_named(daemon, tm, monkeypatch):
    from types import SimpleNamespace

    class _HubHTTPError(Exception):
        response = SimpleNamespace(status_code=429)

    eid = tm.create_episode(tm.get_or_create_task("pick"))
    hf = _FakeHF(fail_at=1, error=_HubHTTPError("429 Too Many Requests"))

    r = await _upload(tm, monkeypatch, hf, [eid])

    assert r["status"] == "error" and r["uploaded"] == []
    assert "rate limit" in r["message"] and "429 Too Many Requests" in r["message"]


def test_the_client_puts_a_whole_batch_in_one_commit(tmp_path, monkeypatch):
    from casquette.fleet import hf as hf_module

    for eid in ("ep1", "ep2"):
        (tmp_path / eid).mkdir()
        (tmp_path / eid / "raw_video.mp4").write_bytes(b"v")
        (tmp_path / eid / "imu_data.json").write_text("{}")
    (tmp_path / "ep1" / ".git").mkdir()
    (tmp_path / "ep1" / ".git" / "HEAD").write_text("x")   # never committed

    class _Api:
        def __init__(self):
            self.commits = []

        def create_repo(self, *a, **k):
            pass

        def create_commit(self, **k):
            self.commits.append(k)

    api = _Api()
    client = hf_module.HuggingFaceClient()
    monkeypatch.setattr(client, "_get_api", lambda: api)

    client.upload_episodes([(tmp_path / "ep1", "ep1/casquette"),
                            (tmp_path / "ep2", "ep2/casquette")],
                           "u/raw", commit_message="batch 1/1 (casquette)")

    assert len(api.commits) == 1
    assert api.commits[0]["commit_message"] == "batch 1/1 (casquette)"
    assert sorted(op.path_in_repo for op in api.commits[0]["operations"]) == [
        f"{eid}/casquette/{name}" for eid in ("ep1", "ep2")
        for name in ("imu_data.json", "raw_video.mp4")]
