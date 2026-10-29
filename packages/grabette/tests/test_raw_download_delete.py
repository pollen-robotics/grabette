"""delete_raw_repos: only this user's raw-download repos are ever deleted."""
import asyncio

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
