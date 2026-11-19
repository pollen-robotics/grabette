"""delete_raw_repos: only this user's raw-download repos are ever deleted."""
import asyncio
import threading

from grabette import hf
from grabette.app import main

USER = "operator"
OURS = f"{USER}/pick-rawdl-20261029-100000-aaaa1111"


class FakeApi:
    def __init__(self):
        self.deleted = []

    def whoami(self):
        return {"name": USER}

    def delete_repo(self, repo, **kw):
        assert kw == {"repo_type": "dataset", "missing_ok": True}
        self.deleted.append(repo)


def _client(monkeypatch):
    api = FakeApi()
    c = hf.HuggingFaceClient()
    monkeypatch.setattr(c, "_get_api", lambda: api)
    return c, api


def test_a_download_repo_of_this_user_is_deleted(monkeypatch):
    c, api = _client(monkeypatch)
    assert c.delete_raw_download_repos([OURS]) == {"deleted": [OURS], "refused": [], "failed": []}
    assert api.deleted == [OURS]


def test_anything_else_is_refused(monkeypatch):
    c, api = _client(monkeypatch)
    others = [
        f"{USER}/my-real-dataset",
        f"{USER}/pick-rawdl-20261029-100000-aaaa1111-backup",
        "pollen-robotics/pick-rawdl-20261029-100000-aaaa1111",
        "someone/pick-rawdl-20261029-100000-aaaa1111",
        "pick-rawdl-20261029-100000-aaaa1111",
    ]
    res = c.delete_raw_download_repos(others)
    assert res["refused"] == others and res["deleted"] == []
    assert api.deleted == []


def test_the_command_reports_a_refusal_as_an_error(monkeypatch):
    c, api = _client(monkeypatch)
    monkeypatch.setattr("grabette.app.routers.huggingface.get_hf_client", lambda: c)
    res = asyncio.run(main._dispatch_relay_command(
        {"type": "delete_raw_repos", "id": "c1", "args": {"repos": [OURS, f"{USER}/my-real-dataset"]}}))
    assert res["status"] == "error"
    assert res["deleted"] == [OURS] and res["refused"] == [f"{USER}/my-real-dataset"]


def test_the_command_needs_no_daemon(monkeypatch):
    c, api = _client(monkeypatch)
    monkeypatch.setattr("grabette.app.routers.huggingface.get_hf_client", lambda: c)
    monkeypatch.setattr(main, "get_daemon_instance", lambda: None)
    res = asyncio.run(main._dispatch_relay_command(
        {"type": "delete_raw_repos", "id": "c2", "args": {"repos": [OURS]}}))
    assert res["status"] == "ok" and api.deleted == [OURS]


# Same cases as grabette-fleet's test_raw_download.py.
FLEET_BUILT = [
    "pick-and-place-rawdl-20261029-100000-aaaa1111",
    "a" * 31 + "-rawdl-20261029-100000-aaaa1111",
]
NOT_A_DOWNLOAD = [
    "pick-rawdl-20261029-100000-aaaa1111-backup",
    "pick-rawdl-20261029-100000",
    "pick-rawdl-20261029-100000-AAAA1111",
    "my-dataset",
]


def test_the_name_pattern_agrees_with_the_fleet():
    for name in FLEET_BUILT:
        assert hf.RAW_DOWNLOAD_NAME_RE.fullmatch(name), name
    for name in NOT_A_DOWNLOAD:
        assert not hf.RAW_DOWNLOAD_NAME_RE.fullmatch(name), name


def test_the_delete_skips_the_relay_queue():
    from grabette.relay_client import _FAST_PATH_TYPES
    assert "delete_raw_repos" in _FAST_PATH_TYPES


def test_a_saturated_upload_pool_does_not_hold_up_the_delete(monkeypatch):
    c, api = _client(monkeypatch)
    monkeypatch.setattr("grabette.app.routers.huggingface.get_hf_client", lambda: c)
    release = threading.Event()
    pool = main._get_upload_executor()
    stuck = [pool.submit(release.wait) for _ in range(main._UPLOAD_WORKERS)]
    try:
        res = asyncio.run(main._dispatch_relay_command(
            {"type": "delete_raw_repos", "id": "c3", "args": {"repos": [OURS]}}))
        assert res["status"] == "ok" and api.deleted == [OURS]
    finally:
        release.set()
        for f in stuck:
            f.result()


def test_a_hung_delete_is_reported_not_waited_on(monkeypatch):
    c, api = _client(monkeypatch)
    release = threading.Event()
    monkeypatch.setattr(api, "delete_repo", lambda repo, **kw: release.wait())
    monkeypatch.setattr("grabette.app.routers.huggingface.get_hf_client", lambda: c)
    monkeypatch.setattr(main, "_DELETE_RAW_TIMEOUT_S", 0.1)
    try:
        res = asyncio.run(main._dispatch_relay_command(
            {"type": "delete_raw_repos", "id": "c4", "args": {"repos": [OURS]}}))
        assert res["status"] == "error" and "timed out" in res["message"]
    finally:
        release.set()
