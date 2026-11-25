"""The JSON endpoints behind the dashboard: tasks/episodes, and the page itself."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from casquette.app.main import create_app


@pytest.fixture
def client(tm):
    # No `with`: the lifespan (daemon, relay) does not run.
    return TestClient(create_app())


def _episode(tm, task_id, eid, files=("raw_video.mp4", "imu_data.json", "frame_timestamps.json"),
             meta=None):
    tm.create_episode(task_id, episode_id=eid)
    d = tm.episode_dir(eid)
    for f in files:
        (d / f).write_text("x")
    (d / "metadata.json").write_text(json.dumps(
        meta or {"duration_seconds": 3.0, "frame_count": 100, "imu_sample_count": 600}))
    tm.register_episode(eid)


def test_dashboard_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "CASQUETTE" in r.text


def test_tasks_list_reports_what_each_episode_lacks(client, tm):
    tid = tm.get_or_create_task("pick cup")
    _episode(tm, tid, "20261120_100000")
    _episode(tm, tid, "20261120_100100", files=("raw_video.mp4",))
    _episode(tm, tid, "20261120_100200", meta={"camera_ok": False})

    task = next(t for t in client.get("/api/tasks").json() if t["name"] == "pick cup")
    by_id = {e["episode_id"]: e for e in task["episodes"]}
    assert by_id["20261120_100000"]["complete"] is True
    assert by_id["20261120_100100"]["missing"] == ["imu_data.json", "frame_timestamps.json"]
    assert by_id["20261120_100200"]["camera_ok"] is False
    assert by_id["20261120_100200"]["complete"] is False


def test_video_still_muxing_is_saving_not_missing(client, tm):
    tid = tm.get_or_create_task("t")
    _episode(tm, tid, "20261120_100000", files=("imu_data.json", "frame_timestamps.json"))
    (tm.episode_dir("20261120_100000") / "raw_video.h264").write_text("x")
    c = client.get("/api/episodes/20261120_100000/check").json()
    assert c["saving"] is True
    assert c["complete"] is False


def test_episode_ids_cannot_leave_the_episodes_dir(client, tm):
    assert client.get("/api/episodes/..%2F..%2Fetc/check").status_code in (400, 404)
    assert client.get("/api/episodes/a.b/video").status_code == 400
    assert client.get("/api/episodes/20990101_000000/check").status_code == 404


def test_delete_episode_and_its_emptied_task(client, tm):
    tid = tm.get_or_create_task("t")
    _episode(tm, tid, "20261120_100000")
    assert client.delete("/api/episodes/20261120_100000").json() == {"deleted": "20261120_100000"}
    assert not tm.episode_dir("20261120_100000").exists()
    assert [t["name"] for t in client.get("/api/tasks").json()] == ["Unassigned"]


def test_episode_being_recorded_cannot_be_deleted(client, tm):
    tm.create_episode(None, episode_id="20261120_100000")   # pending: capture running
    assert client.delete("/api/episodes/20261120_100000").status_code == 409
    assert client.delete("/api/episodes").status_code == 409


def test_delete_all_episodes(client, tm):
    tid = tm.get_or_create_task("t")
    _episode(tm, tid, "20261120_100000")
    _episode(tm, None, "20261120_100100")
    assert client.delete("/api/episodes").json() == {"deleted": 2}
    tasks = client.get("/api/tasks").json()
    assert [(t["name"], t["episode_count"]) for t in tasks] == [("Unassigned", 0)]
    assert list(tm.episodes_dir.iterdir()) == []


def test_download_several_episodes(client, tm):
    tid = tm.get_or_create_task("t")
    _episode(tm, tid, "20261120_100000")
    _episode(tm, tid, "20261120_100100")
    r = client.get("/api/episodes/download",
                   params=[("episode_ids", "20261120_100000"), ("episode_ids", "20261120_100100")])
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/gzip"
