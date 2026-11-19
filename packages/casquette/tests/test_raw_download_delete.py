"""delete_raw_repos on casquette: same guard as grabette's."""

import threading

from casquette.app.main import _handle_relay_command
from casquette.fleet import hf

USER = "operator"
OURS = f"{USER}/pick-rawdl-20261029-100000-aaaa1111"


class FakeApi:
    def __init__(self):
        self.deleted = []

    def whoami(self):
        return {"name": USER}

    def delete_repo(self, repo, **kw):
        self.deleted.append(repo)


async def test_only_this_users_download_repos_are_deleted(monkeypatch):
    api = FakeApi()
    c = hf.HuggingFaceClient()
    monkeypatch.setattr(c, "_get_api", lambda: api)
    monkeypatch.setattr(hf, "get_hf_client", lambda: c)
    res = await _handle_relay_command({"type": "delete_raw_repos", "id": "c1", "args": {
        "repos": [OURS, f"{USER}/my-real-dataset", "someone/pick-rawdl-20261029-100000-aaaa1111"]}})
    assert res["status"] == "error"
    assert res["deleted"] == [OURS]
    assert api.deleted == [OURS]


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
    from casquette.fleet.relay_client import _FAST_PATH_TYPES
    assert "delete_raw_repos" in _FAST_PATH_TYPES


async def test_a_hung_delete_is_reported_not_waited_on(monkeypatch):
    from casquette.app import main

    api = FakeApi()
    release = threading.Event()
    monkeypatch.setattr(api, "delete_repo", lambda repo, **kw: release.wait())
    c = hf.HuggingFaceClient()
    monkeypatch.setattr(c, "_get_api", lambda: api)
    monkeypatch.setattr(hf, "get_hf_client", lambda: c)
    monkeypatch.setattr(main, "_DELETE_RAW_TIMEOUT_S", 0.1)
    try:
        res = await _handle_relay_command(
            {"type": "delete_raw_repos", "id": "c2", "args": {"repos": [OURS]}})
        assert res["status"] == "error" and "timed out" in res["message"]
    finally:
        release.set()
