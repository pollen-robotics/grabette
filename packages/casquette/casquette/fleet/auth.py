# Minimal PAT-only HF auth for the casquette fleet prototype. Grabette's
# auth.py (develop @0cb3453) additionally implements OAuth (PKCE) + refresh-token
# rotation via the fleet Space; we deliberately start with Personal-Access-Token
# only (set once via save_token / `huggingface-cli login` / HF_TOKEN). The full
# OAuth flow can be ported later — this file is the seam where that would slot in.
"""HuggingFace authentication (PAT-only) for the fleet client.

The relay client needs `get_token()` to return a valid HF token; every fleet
request carries it as a Bearer credential. The token is stored with the standard
huggingface_hub mechanism (~/.cache/huggingface/token) so any downstream HF call
picks it up.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import get_token, logout, whoami
from huggingface_hub.errors import HfHubHTTPError

logger = logging.getLogger(__name__)


def _write_secret(path: Path, text: str) -> None:
    """Write the token file with owner-only perms (best-effort chmod 0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    try:
        path.chmod(0o600)
    except OSError:
        pass


class HFAuth:
    """PAT-based HuggingFace login with local persistence."""

    def save_token(self, token: str) -> dict[str, Any]:
        """Validate a token against the HF API and persist it if valid.

        Writes to huggingface_hub's resolved path directly (honors HF_HOME /
        HF_TOKEN_PATH) so it works for both PATs and OAuth tokens.
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
        return True

    async def ensure_authenticated(self) -> bool:
        """A PAT is long-lived — just confirm one is present. (No refresh flow;
        kept async so the lifespan call site matches grabette's, easing a later
        OAuth port.)"""
        return bool(self.get_token())

    def status(self) -> dict[str, Any]:
        token = self.get_token()
        if not token:
            return {"is_logged_in": False, "username": None}
        try:
            info = whoami(token=token)
            return {"is_logged_in": True, "username": info.get("name", "")}
        except Exception:  # noqa: BLE001
            return {"is_logged_in": False, "username": None}


_hf_auth_singleton: Optional[HFAuth] = None


def get_hf_auth() -> HFAuth:
    global _hf_auth_singleton
    if _hf_auth_singleton is None:
        _hf_auth_singleton = HFAuth()
    return _hf_auth_singleton
