"""Hugging Face authentication for casquette: OAuth (PKCE) + manual token.

Ported from grabette/auth.py (same OAuth app, same fleet relay, same refresh
logic), but without importing huggingface_hub: it costs ~11 MB resident on the
512 MB Pi Zero 2W (see casquette.app.main._read_hf_token). HTTP goes through
urllib; callers on the event loop run these in a thread.

The access token is written where huggingface_hub looks for it
(HF_TOKEN_PATH, else HF_HOME/token), so the relay and the uploads pick it up on
their next call. The refresh token sits next to it in casquette-oauth.json.

OAuth callback route: HF redirects to the fleet Space
(``{relay_url}/oauth/grabette/callback``), which 302s the browser to
``http://{hostname}.local:8000/api/hf-auth/oauth/callback`` — the hostname
travels in the state as ``{hostname}|{session_id}``. One redirect URI covers
every device, grabette or casquette, as long as they serve port 8000.

Environment:
    HF_OAUTH_CLIENT_ID   Client id of the HF OAuth app (default: grabette's).
    HF_OAUTH_SCOPES      Space-separated scopes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from casquette.config import settings

logger = logging.getLogger(__name__)

OAUTH_CALLBACK_PATH = "/api/hf-auth/oauth/callback"
# Same OAuth app as grabette (its redirect URI is the fleet relay).
_DEFAULT_OAUTH_CLIENT_ID = "528a5f59-3676-4d5b-8aca-6c5a4db99b42"
OAUTH_CLIENT_ID: Optional[str] = os.environ.get("HF_OAUTH_CLIENT_ID", _DEFAULT_OAUTH_CLIENT_ID)
# write-repos + manage-repos are what dataset create/upload needs.
OAUTH_SCOPES = os.environ.get("HF_OAUTH_SCOPES", "openid profile read-repos write-repos manage-repos")

_HOSTNAME = socket.gethostname()
_RELAY_URL = (settings.relay_url or "").rstrip("/")
if _RELAY_URL:
    OAUTH_REDIRECT_URI = f"{_RELAY_URL}/oauth/grabette/callback"
else:
    # Direct mode (local dev, no relay): HF calls this device back itself.
    OAUTH_REDIRECT_URI = f"http://localhost:{settings.port}{OAUTH_CALLBACK_PATH}"

_OAUTH_SESSION_TTL = 600  # 10 minutes to finish the login in the popup
_TOKEN_ENDPOINT = "https://huggingface.co/oauth/token"
_AUTHORIZE_ENDPOINT = "https://huggingface.co/oauth/authorize"
_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
_WHOAMI_TTL_S = 300.0
# Renew this long before the access token's stated expiry. Larger than the
# refresh loop's period (casquette.app.main) so a tick always lands in the window.
_REFRESH_MARGIN_S = 900


# ── Token storage ──────────────────────────────────────────────────────

def token_path() -> Path:
    """Same resolution as huggingface_hub (HF_TOKEN_PATH, else HF_HOME/token)."""
    path = os.environ.get("HF_TOKEN_PATH")
    if path:
        return Path(path)
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        hf_home = os.path.join(cache, "huggingface")
    return Path(hf_home) / "token"


def read_token() -> Optional[str]:
    """HF_TOKEN env var, else the token file — like huggingface_hub.get_token()."""
    tok = os.environ.get("HF_TOKEN")
    if tok and tok.strip():
        return tok.strip()
    try:
        return token_path().read_text().strip() or None
    except OSError:
        return None


def _write_secret(path: Path, text: str) -> None:
    """Write a credential file with owner-only perms (best-effort chmod 0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _oauth_store_path() -> Path:
    return token_path().with_name("casquette-oauth.json")


# ── HTTP helpers ───────────────────────────────────────────────────────

class HubRefused(Exception):
    """The Hub answered, and said no (bad token, dead refresh token…)."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status = status


def _post_form(url: str, data: dict[str, str]) -> dict[str, Any]:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        # Never log the body: the token endpoint can echo request details.
        raise HubRefused(e.code) from None


@dataclass
class OAuthSession:
    session_id: str
    code_verifier: str
    status: str = "pending"  # pending | authorized | error
    username: Optional[str] = None
    error_message: Optional[str] = None
    expires_at: float = field(default_factory=lambda: time.time() + _OAUTH_SESSION_TTL)


class HFAuth:
    """HF login: manual token + OAuth (PKCE), with refresh-token renewal."""

    def __init__(self, client_id: Optional[str] = OAUTH_CLIENT_ID, scopes: str = OAUTH_SCOPES,
                 redirect_uri: str = OAUTH_REDIRECT_URI) -> None:
        self.client_id = client_id
        self.scopes = scopes
        self.redirect_uri = redirect_uri
        self._sessions: dict[str, OAuthSession] = {}
        # token → (checked_at, username or None when the Hub refused it)
        self._whoami_cache: dict[str, tuple[float, Optional[str]]] = {}

    # ── who is this token ─────────────────────────────────────────────

    def whoami(self, token: str) -> Optional[str]:
        """Username for this token, None if the Hub refuses it.
        Raises OSError when the Hub cannot be reached."""
        cached = self._whoami_cache.get(token)
        if cached and time.monotonic() - cached[0] < _WHOAMI_TTL_S:
            return cached[1]
        req = urllib.request.Request(_WHOAMI_URL, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                name: Optional[str] = json.load(r).get("name") or ""
        except urllib.error.HTTPError as e:
            if e.code not in (401, 403):
                raise
            name = None
        self._whoami_cache[token] = (time.monotonic(), name)
        return name

    def status(self) -> dict[str, Any]:
        token = read_token()
        from_env = bool((os.environ.get("HF_TOKEN") or "").strip())
        if not token:
            return {"is_logged_in": False, "username": None, "from_env": False}
        try:
            name = self.whoami(token)
        except OSError:
            # Offline: a token is there, it just can't be checked right now.
            return {"is_logged_in": True, "username": None, "offline": True, "from_env": from_env}
        return {"is_logged_in": name is not None, "username": name, "from_env": from_env}

    # ── manual token ──────────────────────────────────────────────────

    def save_token(self, token: str) -> dict[str, Any]:
        """Validate a token and persist it. A manual token replaces any OAuth
        login, so the refresh store goes: it would overwrite this token."""
        token = token.strip()
        if not token:
            return {"status": "error", "message": "Empty token"}
        try:
            name = self.whoami(token)
        except OSError:
            return {"status": "error", "message": "Hugging Face is unreachable from the device"}
        if name is None:
            return {"status": "error", "message": "Hugging Face refused this token"}
        _write_secret(token_path(), token)
        _oauth_store_path().unlink(missing_ok=True)
        return {"status": "success", "username": name}

    def delete_token(self) -> None:
        """Log out: the token AND the refresh token, or the next refresh would
        silently log the device back in."""
        token_path().unlink(missing_ok=True)
        _oauth_store_path().unlink(missing_ok=True)
        self._whoami_cache.clear()

    # ── refresh token ─────────────────────────────────────────────────

    def _load_oauth_store(self) -> dict[str, Any]:
        try:
            return json.loads(_oauth_store_path().read_text())
        except (OSError, ValueError):
            return {}

    def _save_oauth_store(self, token_data: dict[str, Any]) -> None:
        """Keep the newest refresh token (HF rotates it) and the access token's
        expiry. No refresh token in the response → keep the existing store."""
        refresh = token_data.get("refresh_token")
        if not refresh:
            logger.warning("HF token response had no refresh_token; the device will "
                           "need a new login once the access token expires")
            return
        expires_in = token_data.get("expires_in")
        _write_secret(_oauth_store_path(), json.dumps({
            "refresh_token": refresh,
            "expires_at": time.time() + float(expires_in) if expires_in else None,
        }))

    def refresh_access_token(self) -> bool:
        refresh = self._load_oauth_store().get("refresh_token")
        if not refresh or not self.client_id:
            return False
        try:
            token_data = _post_form(_TOKEN_ENDPOINT, {
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": refresh,
            })
        except HubRefused as e:
            # 400/401: the refresh token is dead (revoked/expired) — drop it so
            # we stop retrying; a new login is needed.
            if e.status in (400, 401):
                _oauth_store_path().unlink(missing_ok=True)
            logger.warning("HF token refresh refused (%s)", e)
            return False
        except OSError as e:
            logger.info("HF token refresh failed (offline?): %s", e)
            return False
        access = token_data.get("access_token") or token_data.get("accessToken")
        if not access:
            return False
        _write_secret(token_path(), access)
        self._save_oauth_store(token_data)
        logger.info("refreshed HF access token")
        return True

    def ensure_authenticated(self) -> bool:
        """Renew the access token when it is missing or about to expire. Called
        at startup and periodically (blocking: run it in a thread). A manual
        token (no refresh store) is trusted as is."""
        store = self._load_oauth_store()
        if not store.get("refresh_token"):
            return bool(read_token())
        exp = store.get("expires_at")
        near_expiry = exp is not None and time.time() >= exp - _REFRESH_MARGIN_S
        if not read_token() or near_expiry:
            return self.refresh_access_token()
        return True

    # ── OAuth (PKCE) ──────────────────────────────────────────────────

    def oauth_configured(self) -> bool:
        return bool(self.client_id)

    def warm_relay(self, timeout_s: float = 45.0) -> str:
        """Wake the fleet Space before the login: HF sends the callback through
        it, and a sleeping free-tier Space would drop it. Returns ok | timeout | skipped."""
        if not _RELAY_URL:
            return "skipped"
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{_RELAY_URL}/healthz", timeout=10):
                    return "ok"
            except urllib.error.HTTPError as e:
                if e.code < 500:  # any real answer ⇒ the Space is up
                    return "ok"
            except OSError:
                pass  # still waking
            time.sleep(2)
        return "timeout"

    def _cleanup_sessions(self) -> None:
        now = time.time()
        for sid in [s for s, v in self._sessions.items() if v.expires_at < now]:
            del self._sessions[sid]

    def start_oauth(self) -> dict[str, Any]:
        """Open a session; returns the HF authorization URL for the popup."""
        self._cleanup_sessions()
        if not self.client_id:
            return {"status": "error", "message": "OAuth not configured. Set HF_OAUTH_CLIENT_ID."}
        session_id = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(32)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        self._sessions[session_id] = OAuthSession(session_id=session_id, code_verifier=verifier)
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.scopes,
            "response_type": "code",
            # The relay reads the hostname to send the browser back to this device.
            "state": f"{_HOSTNAME}|{session_id}" if _RELAY_URL else session_id,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return {"status": "success", "session_id": session_id,
                "auth_url": f"{_AUTHORIZE_ENDPOINT}?{urllib.parse.urlencode(params)}",
                "expires_in": _OAUTH_SESSION_TTL}

    def oauth_status(self, session_id: str) -> dict[str, Any]:
        self._cleanup_sessions()
        s = self._sessions.get(session_id)
        if not s:
            return {"status": "expired", "message": "Login expired, try again."}
        out: dict[str, Any] = {"status": s.status}
        if s.status == "authorized":
            out["username"] = s.username
        elif s.status == "error":
            out["message"] = s.error_message
        return out

    def fail_session(self, session_id: str, message: str) -> None:
        if s := self._sessions.get(session_id):
            s.status, s.error_message = "error", message

    def exchange_code(self, code: str, session_id: str) -> dict[str, Any]:
        """Trade the authorization code for tokens and persist them."""
        self._cleanup_sessions()
        s = self._sessions.get(session_id)
        if not s:
            return {"status": "error", "message": "Invalid or expired login session."}

        def fail(message: str) -> dict[str, Any]:
            s.status, s.error_message = "error", message
            return {"status": "error", "message": message}

        try:
            token_data = _post_form(_TOKEN_ENDPOINT, {
                "grant_type": "authorization_code",
                "client_id": self.client_id or "",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "code_verifier": s.code_verifier,
            })
        except HubRefused as e:
            return fail(f"Token exchange failed ({e}).")
        except OSError as e:
            return fail(f"Hugging Face is unreachable from the device: {e}")
        access = token_data.get("access_token") or token_data.get("accessToken")
        if not access:
            return fail("No access token in the authorization response.")

        _write_secret(token_path(), access)
        self._save_oauth_store(token_data)
        try:
            username = self.whoami(access) or ""
        except OSError:
            username = ""
        s.status, s.username = "authorized", username
        return {"status": "success", "username": username}


_hf_auth_singleton: Optional[HFAuth] = None


def get_hf_auth() -> HFAuth:
    """Process-wide: the router and the refresh loop share sessions and caches."""
    global _hf_auth_singleton
    if _hf_auth_singleton is None:
        _hf_auth_singleton = HFAuth()
    return _hf_auth_singleton
