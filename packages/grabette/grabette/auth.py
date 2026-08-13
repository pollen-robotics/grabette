"""HuggingFace authentication — OAuth (PKCE) + manual token, ported from reachy_mini.

Stripped of the reachy-specific central signaling relay and wireless/Lite daemon
state. The token is persisted with the standard ``huggingface_hub`` mechanism
(``~/.cache/huggingface/token``) so any downstream HF call picks it up.

Config is read from the environment so the team can point it at their own HF
OAuth app:

    GRABETTE_RELAY_URL   URL of the grabette-fleet Space acting as OAuth relay.
                         Defaults to the Pollen fleet Space — no config needed on Pi.
                         Set to empty string to disable relay (e.g. for local dev).
    GRABETTE_BASE_URL    Direct public URL of this grabette (overrides auto-detect).
                         Only needed when not using the relay (e.g. local dev).
                         (default without relay: http://localhost:8000)
    HF_OAUTH_CLIENT_ID   Client id of YOUR HF OAuth app.
    HF_OAUTH_SCOPES      Space-separated scopes. For dataset upload you need at
                         least ``write-repos`` (and ``manage-repos`` to create).

Register an app at https://huggingface.co/settings/connected-applications.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp
from huggingface_hub import get_token, logout, whoami
from huggingface_hub.errors import HfHubHTTPError

# --- configuration ----------------------------------------------------------
# The hostname is encoded into the OAuth state when using the relay, so the
# relay knows which grabette to forward the callback to.
logger = logging.getLogger(__name__)


def _write_secret(path: Path, text: str) -> None:
    """Write a credential file (HF token / refresh store) with owner-only perms.
    chmod 0600 so another local user/process on the device can't read the token;
    best-effort (a no-op on non-POSIX filesystems, which is acceptable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    try:
        path.chmod(0o600)
    except OSError:
        pass

_HOSTNAME = socket.gethostname()

_DEFAULT_RELAY_URL = "https://glannuzel-fleet-test.hf.space"
_RELAY_URL = os.environ.get("GRABETTE_RELAY_URL", _DEFAULT_RELAY_URL).rstrip("/")
OAUTH_CALLBACK_PATH = "/api/hf-auth/oauth/callback"

if _RELAY_URL:
    # Relay mode: redirect_uri points to the fleet Space (one registered URL
    # for all grabettes). The hostname is encoded into the state parameter.
    OAUTH_REDIRECT_URI = f"{_RELAY_URL}/oauth/grabette/callback"
    BASE_URL = os.environ.get("GRABETTE_BASE_URL", f"http://{_HOSTNAME}.local:8000").rstrip("/")
else:
    # Direct mode: for local dev or a grabette with a known public URL.
    BASE_URL = os.environ.get("GRABETTE_BASE_URL", "http://localhost:8000").rstrip("/")
    OAUTH_REDIRECT_URI = f"{BASE_URL}{OAUTH_CALLBACK_PATH}"

# To replace - currently not from Pollen org
_DEFAULT_OAUTH_CLIENT_ID = "528a5f59-3676-4d5b-8aca-6c5a4db99b42"
OAUTH_CLIENT_ID: Optional[str] = os.environ.get(
    "HF_OAUTH_CLIENT_ID", _DEFAULT_OAUTH_CLIENT_ID
)
# write-repos + manage-repos are what dataset create/upload needs.
OAUTH_SCOPES = os.environ.get(
    "HF_OAUTH_SCOPES", "openid profile read-repos write-repos manage-repos"
)

_OAUTH_SESSION_TTL = 600  # 10 minutes
_TOKEN_ENDPOINT = "https://huggingface.co/oauth/token"
# Renew this many seconds before the access token's stated expiry, so a call
# never rides an about-to-expire token. Kept larger than the device's periodic
# refresh interval so at least one tick lands inside the window before expiry.
_REFRESH_MARGIN_S = 900


@dataclass
class OAuthSession:
    """An in-progress OAuth authorization."""

    session_id: str
    state: str  # CSRF protection; doubles as the session id
    code_verifier: str  # PKCE code verifier
    status: str = "pending"  # pending | authorized | error | expired
    access_token: Optional[str] = None
    username: Optional[str] = None
    error_message: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + _OAUTH_SESSION_TTL)


class HFAuth:
    """HuggingFace login: manual token + OAuth (PKCE), with local persistence."""

    def __init__(
        self,
        client_id: Optional[str] = OAUTH_CLIENT_ID,
        scopes: str = OAUTH_SCOPES,
        redirect_uri: str = OAUTH_REDIRECT_URI,
    ) -> None:
        self.client_id = client_id
        self.scopes = scopes
        self.redirect_uri = redirect_uri
        self._sessions: dict[str, OAuthSession] = {}

    # ---- manual token -----------------------------------------------------

    def save_token(self, token: str) -> dict[str, Any]:
        """Validate a token against the HF API and persist it if valid.

        Writes the token to huggingface_hub's resolved path directly (instead of
        login()) so it works for both Personal Access Tokens and OAuth tokens —
        login() raises on OAuth tokens. Honors HF_HOME / HF_TOKEN_PATH.
        """
        try:
            user_info = whoami(token=token)  # validates; raises on invalid token
            from huggingface_hub.constants import HF_TOKEN_PATH

            _write_secret(Path(HF_TOKEN_PATH), token)
            return {"status": "success", "username": user_info.get("name", "")}
        except (HfHubHTTPError, ValueError):
            return {"status": "error", "message": "Invalid token or network error"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": str(e)}

    def get_token(self) -> Optional[str]:
        return get_token()

    def delete_token(self) -> bool:
        try:
            logout()
        except Exception:  # noqa: BLE001
            return False
        # Forget the refresh token too — an explicit logout must not silently
        # re-authenticate the device on the next start.
        try:
            self._oauth_store_path().unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        return True

    # ---- refresh token (stay logged in across restarts / past expiry) -------

    @staticmethod
    def _oauth_store_path() -> Path:
        """Sidecar file (next to the HF token) holding the refresh token +
        access-token expiry. Kept beside HF_TOKEN_PATH so it honors HF_HOME."""
        from huggingface_hub.constants import HF_TOKEN_PATH

        return Path(HF_TOKEN_PATH).with_name("grabette-oauth.json")

    def _load_oauth_store(self) -> dict[str, Any]:
        try:
            return json.loads(self._oauth_store_path().read_text())
        except Exception:  # noqa: BLE001 — missing/corrupt → treated as empty
            return {}

    def _save_oauth_store(self, token_data: dict[str, Any]) -> None:
        """Persist the refresh token + computed access-token expiry from an HF
        token response. No-op (leaves any existing store) if HF returned no
        refresh token, so we never lose a good one."""
        refresh = token_data.get("refresh_token")
        if not refresh:
            logger.warning("HF token response had no refresh_token; device will "
                           "need a manual re-login once the access token expires")
            return
        expires_in = token_data.get("expires_in")
        store = {
            "refresh_token": refresh,
            "expires_at": time.time() + float(expires_in) if expires_in else None,
        }
        try:
            _write_secret(self._oauth_store_path(), json.dumps(store))
        except Exception:  # noqa: BLE001
            logger.warning("failed to persist OAuth refresh store", exc_info=True)

    async def refresh_access_token(self) -> dict[str, Any]:
        """Mint a fresh access token from the stored refresh token and persist
        it (rotating the refresh token, which HF may replace on each use)."""
        store = self._load_oauth_store()
        refresh = store.get("refresh_token")
        if not refresh:
            return {"status": "error", "message": "no refresh token"}
        if not self.client_id:
            return {"status": "error", "message": "OAuth not configured"}
        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "refresh_token": refresh,
        }
        try:
            async with aiohttp.ClientSession() as http:
                async with http.post(_TOKEN_ENDPOINT, data=data) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        # A 400/401 means the refresh token is dead (revoked or
                        # expired) — drop it so we stop retrying and fall back to
                        # a manual login.
                        if resp.status in (400, 401):
                            self._oauth_store_path().unlink(missing_ok=True)
                        return {"status": "error", "message": f"refresh failed (HTTP {resp.status})"}
                    token_data = json.loads(body)
            access_token = token_data.get("access_token") or token_data.get("accessToken")
            if not access_token:
                return {"status": "error", "message": "no access token in refresh response"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": f"{type(e).__name__}: {e}"}

        from huggingface_hub.constants import HF_TOKEN_PATH

        _write_secret(Path(HF_TOKEN_PATH), access_token)
        # HF rotates the refresh token; keep the newest. token_data carries the
        # new refresh_token + expires_in, so reuse the same persistence path.
        self._save_oauth_store(token_data)
        logger.info("refreshed HF access token from stored refresh token")
        return {"status": "success"}

    async def ensure_authenticated(self) -> bool:
        """Best-effort: make sure a usable access token is in place, refreshing
        it from the stored refresh token when it's missing or (near-)expired.
        Called at startup and periodically so the device stays logged in as the
        last account without the operator re-doing OAuth. Returns True if a
        valid token is available afterwards."""
        store = self._load_oauth_store()
        exp = store.get("expires_at")
        near_expiry = exp is not None and time.time() >= exp - _REFRESH_MARGIN_S
        # Refresh when we have no token, or the one we have is about to expire.
        if store.get("refresh_token") and (not self.get_token() or near_expiry):
            if (await self.refresh_access_token()).get("status") == "success":
                return True
        token = self.get_token()
        if not token:
            return False
        # A manual PAT (no refresh token) is long-lived — trust it without a
        # network round-trip. Only validate/refresh OAuth tokens.
        if not store.get("refresh_token"):
            return True
        try:
            whoami(token=token)
            return True
        except Exception:  # noqa: BLE001 — token rejected → try one refresh
            return (await self.refresh_access_token()).get("status") == "success"

    def status(self) -> dict[str, Any]:
        token = self.get_token()
        if not token:
            return {"is_logged_in": False, "username": None}
        try:
            info = whoami(token=token)
            return {"is_logged_in": True, "username": info.get("name", "")}
        except Exception:  # noqa: BLE001
            return {"is_logged_in": False, "username": None}

    # ---- OAuth (PKCE) -----------------------------------------------------

    def oauth_configured(self) -> bool:
        return bool(self.client_id)

    async def warm_relay(self, timeout_s: float = 45.0) -> dict[str, Any]:
        """Wake the fleet Space and wait until it responds, before starting OAuth.

        HF routes the OAuth callback through the Space (the registered
        redirect_uri). A free-tier Space sleeps when idle, and an un-authenticated
        device doesn't poll it (poll needs a token) — so nothing keeps it warm and
        the callback would hit a sleeping Space. Pinging /healthz here wakes it and
        blocks until it's up, so the login works without a dashboard tab open.
        Best-effort: returns {"status": "ok" | "timeout" | "skipped"}."""
        if not _RELAY_URL:
            return {"status": "skipped"}  # direct mode: no relay to wake
        url = f"{_RELAY_URL}/healthz"
        deadline = time.monotonic() + timeout_s
        try:
            async with aiohttp.ClientSession() as http:
                while time.monotonic() < deadline:
                    try:
                        async with http.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                            if r.status < 500:  # any real response ⇒ Space is up
                                return {"status": "ok"}
                    except Exception:  # noqa: BLE001 — still waking / transient
                        pass
                    await asyncio.sleep(2)
        except Exception:  # noqa: BLE001
            pass
        return {"status": "timeout"}

    def _cleanup_sessions(self) -> None:
        now = time.time()
        for sid in [s for s, v in self._sessions.items() if v.expires_at < now]:
            del self._sessions[sid]

    @staticmethod
    def _pkce_pair() -> tuple[str, str]:
        code_verifier = secrets.token_urlsafe(32)
        digest = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        return code_verifier, code_challenge

    def start_oauth(self) -> dict[str, Any]:
        """Create a session and return the HF authorization URL to open."""
        self._cleanup_sessions()
        if not self.client_id:
            return {"status": "error", "message": "OAuth not configured. Set HF_OAUTH_CLIENT_ID."}

        session_id = secrets.token_urlsafe(32)
        code_verifier, code_challenge = self._pkce_pair()
        self._sessions[session_id] = OAuthSession(
            session_id=session_id, state=session_id, code_verifier=code_verifier
        )
        # When using the relay, encode the grabette hostname into the state so
        # the relay knows where to forward the callback. The grabette uses only
        # the session_id part to look up the session on return.
        oauth_state = f"{_HOSTNAME}|{session_id}" if _RELAY_URL else session_id
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.scopes,
            "response_type": "code",
            "state": oauth_state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return {
            "status": "success",
            "session_id": session_id,
            "auth_url": f"https://huggingface.co/oauth/authorize?{urlencode(params)}",
            "expires_in": _OAUTH_SESSION_TTL,
        }

    def oauth_status(self, session_id: str) -> dict[str, Any]:
        """Poll an OAuth session (frontend polls until authorized)."""
        self._cleanup_sessions()
        session = self._sessions.get(session_id)
        if not session:
            return {"status": "expired", "message": "Session expired or not found"}
        result: dict[str, Any] = {"status": session.status}
        if session.status == "authorized":
            result["username"] = session.username
        elif session.status == "error":
            result["message"] = session.error_message
        return result

    async def exchange_code(self, code: str, state: str) -> dict[str, Any]:
        """Exchange an authorization code for a token, then persist it."""
        self._cleanup_sessions()
        session = self._sessions.get(state)
        if not session:
            return {"status": "error", "message": "Invalid or expired session."}
        if not self.client_id:
            session.status = "error"
            session.error_message = "OAuth not configured"
            return {"status": "error", "message": session.error_message}

        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": session.code_verifier,  # PKCE verification
        }
        try:
            async with aiohttp.ClientSession() as http:
                async with http.post("https://huggingface.co/oauth/token", data=data) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        # Log the raw body server-side (debug) but never return it —
                        # the token endpoint's response can echo request details.
                        logger.debug("OAuth token exchange failed (HTTP %s): %s", resp.status, body)
                        session.status = "error"
                        session.error_message = f"Token exchange failed (HTTP {resp.status})."
                        return {"status": "error", "message": session.error_message}
                    token_data = json.loads(body)
            access_token = token_data.get("access_token") or token_data.get("accessToken")
            if not access_token:
                # Log only the response KEYS (not values) for diagnosis.
                logger.debug("OAuth token response had no access_token; keys=%s", list(token_data))
                session.status = "error"
                session.error_message = "No access token in the authorization response."
                return {"status": "error", "message": session.error_message}
        except Exception as e:  # noqa: BLE001
            session.status = "error"
            session.error_message = f"Token request error: {type(e).__name__}: {e}"
            return {"status": "error", "message": session.error_message}

        # OAuth tokens are written to the standard HF token file directly
        # (login() is finicky with them). Use huggingface_hub's resolved path so
        # we honor HF_HOME / HF_TOKEN_PATH instead of hardcoding ~/.cache.
        try:
            from huggingface_hub.constants import HF_TOKEN_PATH

            _write_secret(Path(HF_TOKEN_PATH), access_token)
        except Exception as e:  # noqa: BLE001
            session.status = "error"
            session.error_message = f"Failed to save token: {type(e).__name__}: {e}"
            return {"status": "error", "message": session.error_message}

        # Persist the refresh token so the device can renew the (short-lived)
        # access token on its own — staying logged in as this account across
        # restarts and past expiry, without the operator re-doing OAuth. HF may
        # rotate the refresh token on each use, so we always store the latest.
        self._save_oauth_store(token_data)

        username = ""
        try:
            info = whoami(token=access_token)
            if isinstance(info, dict):
                username = info.get("name", "") or info.get("fullname", "")
        except Exception:  # noqa: BLE001
            pass

        session.status = "authorized"
        session.access_token = access_token
        session.username = username
        return {"status": "success", "username": username}


# Process-wide singleton so the app factory (auth router) and the startup
# lifespan (token refresh) share the same HFAuth — in particular the same
# in-progress OAuth sessions and refresh-token handling.
_hf_auth_singleton: Optional[HFAuth] = None


def get_hf_auth() -> HFAuth:
    global _hf_auth_singleton
    if _hf_auth_singleton is None:
        _hf_auth_singleton = HFAuth()
    return _hf_auth_singleton
