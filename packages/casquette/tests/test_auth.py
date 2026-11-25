"""HF login: OAuth (PKCE) through the fleet relay, refresh, manual token, logout.

The Hub is never called: _post_form (token endpoint) and whoami are stubbed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import casquette.fleet.auth as auth_mod
from casquette.app.main import create_app


@pytest.fixture
def auth(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    monkeypatch.setattr(auth_mod, "_RELAY_URL", "https://fleet.example")
    monkeypatch.setattr(auth_mod, "_HOSTNAME", "casquette-test")
    a = auth_mod.HFAuth(client_id="cid", redirect_uri="https://fleet.example/oauth/grabette/callback")
    monkeypatch.setattr(a, "whoami", lambda token: None if token == "bad" else "alice")
    monkeypatch.setattr(auth_mod, "_hf_auth_singleton", a)
    return a


def _store(tmp_path):
    p = tmp_path / "casquette-oauth.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_oauth_start_routes_the_callback_through_the_relay(auth):
    r = auth.start_oauth()
    q = parse_qs(urlparse(r["auth_url"]).query)
    assert q["redirect_uri"] == ["https://fleet.example/oauth/grabette/callback"]
    # The relay splits "{hostname}|{session}" to send the browser back here.
    assert q["state"] == [f"casquette-test|{r['session_id']}"]
    verifier = auth._sessions[r["session_id"]].code_verifier
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert q["code_challenge"] == [expected]


def test_oauth_callback_saves_token_and_refresh_token(auth, tmp_path, monkeypatch):
    sent = {}

    def fake_post(url, data):
        sent.update(data)
        return {"access_token": "at1", "refresh_token": "rt1", "expires_in": 3600}

    monkeypatch.setattr(auth_mod, "_post_form", fake_post)
    sid = auth.start_oauth()["session_id"]
    client = TestClient(create_app())
    r = client.get("/api/hf-auth/oauth/callback", params={"code": "c0de", "state": sid})
    assert r.status_code == 200 and "Signed in" in r.text
    assert sent["code_verifier"] == auth._sessions[sid].code_verifier
    assert (tmp_path / "token").read_text() == "at1"
    assert _store(tmp_path)["refresh_token"] == "rt1"
    assert client.get(f"/api/hf-auth/oauth/status/{sid}").json() == {
        "status": "authorized", "username": "alice"}


def test_oauth_error_from_hf_fails_the_session(auth):
    sid = auth.start_oauth()["session_id"]
    client = TestClient(create_app())
    r = client.get("/api/hf-auth/oauth/callback",
                   params={"error": "access_denied", "state": sid})
    assert "failed" in r.text
    assert client.get(f"/api/hf-auth/oauth/status/{sid}").json()["status"] == "error"


def test_unknown_session_is_refused(auth, monkeypatch):
    monkeypatch.setattr(auth_mod, "_post_form", lambda *a: pytest.fail("must not call the Hub"))
    assert auth.exchange_code("c0de", "nope")["status"] == "error"


def test_refresh_renews_a_token_near_expiry(auth, tmp_path, monkeypatch):
    (tmp_path / "token").write_text("old")
    (tmp_path / "casquette-oauth.json").write_text(
        json.dumps({"refresh_token": "rt1", "expires_at": time.time() + 60}))
    monkeypatch.setattr(auth_mod, "_post_form",
                        lambda url, data: {"access_token": "new", "refresh_token": "rt2", "expires_in": 3600})
    assert auth.ensure_authenticated() is True
    assert (tmp_path / "token").read_text() == "new"
    assert _store(tmp_path)["refresh_token"] == "rt2"   # rotated


def test_fresh_token_is_not_refreshed(auth, tmp_path, monkeypatch):
    (tmp_path / "token").write_text("at")
    (tmp_path / "casquette-oauth.json").write_text(
        json.dumps({"refresh_token": "rt1", "expires_at": time.time() + 3600}))
    monkeypatch.setattr(auth_mod, "_post_form", lambda *a: pytest.fail("no refresh needed"))
    assert auth.ensure_authenticated() is True


def test_dead_refresh_token_is_dropped(auth, tmp_path, monkeypatch):
    (tmp_path / "casquette-oauth.json").write_text(json.dumps({"refresh_token": "rt1", "expires_at": 0}))

    def refused(url, data):
        raise auth_mod.HubRefused(400)

    monkeypatch.setattr(auth_mod, "_post_form", refused)
    assert auth.ensure_authenticated() is False
    assert _store(tmp_path) is None


def test_manual_token_replaces_an_oauth_login(auth, tmp_path):
    (tmp_path / "casquette-oauth.json").write_text(json.dumps({"refresh_token": "rt1"}))
    client = TestClient(create_app())
    assert client.post("/api/hf-auth/save-token", json={"token": "bad"}).status_code == 400
    r = client.post("/api/hf-auth/save-token", json={"token": "hf_good"})
    assert r.json() == {"is_logged_in": True, "username": "alice"}
    assert (tmp_path / "token").read_text() == "hf_good"
    assert _store(tmp_path) is None   # the refresh would overwrite the manual token


def test_logout_forgets_the_refresh_token_too(auth, tmp_path):
    (tmp_path / "token").write_text("at")
    (tmp_path / "casquette-oauth.json").write_text(json.dumps({"refresh_token": "rt1"}))
    client = TestClient(create_app())
    assert client.delete("/api/hf-auth/token").json() == {"is_logged_in": False}
    assert not (tmp_path / "token").exists()
    assert _store(tmp_path) is None
    assert client.get("/api/hf-auth/status").json()["is_logged_in"] is False
