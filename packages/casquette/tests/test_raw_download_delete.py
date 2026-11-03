"""delete_raw_repos on casquette: same guard as grabette's."""

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
