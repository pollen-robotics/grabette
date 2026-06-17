"""HuggingFace authentication — OAuth (PKCE) + manual token, ported from reachy_mini.

Stripped of the reachy-specific central signaling relay and wireless/Lite daemon
state. The token is persisted with the standard ``huggingface_hub`` mechanism
(``~/.cache/huggingface/token``) so any downstream HF call picks it up.

Config is read from the environment so the team can point it at their own HF
OAuth app:

    GRABETTE_BASE_URL    Public URL this dashboard is reachable at.
                         The OAuth callback is appended to it and the resulting
                         URL must be registered with the HF OAuth app.
                         (default: http://localhost:8000)
    HF_OAUTH_CLIENT_ID   Client id of YOUR HF OAuth app.
    HF_OAUTH_SCOPES      Space-separated scopes. For dataset upload you need at
                         least ``write-repos`` (and ``manage-repos`` to create).

Register an app at https://huggingface.co/settings/connected-applications.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp
from huggingface_hub import get_token, logout, whoami
from huggingface_hub.errors import HfHubHTTPError

# --- configuration ----------------------------------------------------------
BASE_URL = os.environ.get("GRABETTE_BASE_URL", "http://localhost:8000").rstrip("/")
OAUTH_CALLBACK_PATH = "/api/hf-auth/oauth/callback"
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

            token_path = Path(HF_TOKEN_PATH)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(token)
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
            return True
        except Exception:  # noqa: BLE001
            return False

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

        state = secrets.token_urlsafe(32)
        code_verifier, code_challenge = self._pkce_pair()
        self._sessions[state] = OAuthSession(
            session_id=state, state=state, code_verifier=code_verifier
        )
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.scopes,
            "response_type": "code",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        return {
            "status": "success",
            "session_id": state,
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

    def cancel_oauth(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

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
                        session.status = "error"
                        session.error_message = f"Token exchange failed (HTTP {resp.status}): {body}"
                        return {"status": "error", "message": session.error_message}
                    token_data = json.loads(body)
            access_token = token_data.get("access_token") or token_data.get("accessToken")
            if not access_token:
                session.status = "error"
                session.error_message = f"No access token. Response: {token_data}"
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

            token_path = Path(HF_TOKEN_PATH)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(access_token)
        except Exception as e:  # noqa: BLE001
            session.status = "error"
            session.error_message = f"Failed to save token: {type(e).__name__}: {e}"
            return {"status": "error", "message": session.error_message}

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
