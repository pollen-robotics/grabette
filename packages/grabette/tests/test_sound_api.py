"""The /api/sound endpoints behind the dashboard's volume control."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from grabette.app.routers import sound as sound_router
from grabette.hardware import sound


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setattr(sound, "VOLUME_FILE", tmp_path / "sound_volume")
    speaker = sound.Speaker(enabled=False, volume=0.6)
    monkeypatch.setattr(sound, "_speaker", speaker)
    monkeypatch.setattr(sound_router, "_is_capturing", lambda: False)
    app = FastAPI()
    app.include_router(sound_router.router)
    return TestClient(app), speaker


def test_volume_is_reported_in_percent(api):
    client, _ = api
    r = client.get("/api/sound")
    assert r.status_code == 200
    assert r.json()["volume"] == 60
    assert r.json()["available"] is False


def test_set_volume_applies_and_persists(api):
    client, speaker = api
    r = client.put("/api/sound/volume", json={"volume": 25})
    assert r.status_code == 200
    assert r.json()["volume"] == 25
    assert speaker.volume == 0.25
    assert sound.load_saved_volume(0.6) == 0.25


def test_volume_out_of_range_is_refused(api):
    client, _ = api
    assert client.put("/api/sound/volume", json={"volume": 101}).status_code == 422
    assert client.put("/api/sound/volume", json={"volume": -1}).status_code == 422


def test_test_is_refused_without_a_speaker(api):
    client, _ = api
    r = client.post("/api/sound/test")
    assert r.status_code == 409


def test_test_is_refused_during_a_recording(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(sound_router, "_is_capturing", lambda: True)
    r = client.post("/api/sound/test")
    assert r.status_code == 409
    assert "recording" in r.json()["detail"]


def test_status_says_whether_a_test_is_playing(api):
    client, speaker = api
    assert client.get("/api/sound").json()["testing"] is False
    speaker._testing = True
    assert client.get("/api/sound").json()["testing"] is True


def test_test_of_an_unknown_cue_is_refused(api):
    client, _ = api
    r = client.post("/api/sound/test", json={"cue": "volume_preview"})
    assert r.status_code == 422
