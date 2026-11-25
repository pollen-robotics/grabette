"""Hugging Face account for the dashboard: status, OAuth login, manual token, logout.

Same routes as grabette's (/api/hf-auth/...) — in particular the OAuth callback
path, which the fleet relay redirects the browser to. The logic lives in
casquette.fleet.auth; blocking calls (HTTP to the Hub) run in FastAPI's
threadpool via plain `def` endpoints.
"""

from __future__ import annotations

import html

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from casquette.fleet.auth import get_hf_auth

router = APIRouter(prefix="/api/hf-auth", tags=["hf-auth"])


class TokenRequest(BaseModel):
    token: str


@router.get("/status")
def status():
    return get_hf_auth().status()


@router.post("/save-token")
def save_token(req: TokenRequest):
    result = get_hf_auth().save_token(req.token)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return {"is_logged_in": True, "username": result["username"]}


@router.delete("/token")
def delete_token():
    get_hf_auth().delete_token()
    return {"is_logged_in": False}


@router.get("/oauth/configured")
def oauth_configured():
    return {"configured": get_hf_auth().oauth_configured()}


@router.get("/oauth/warm-relay")
def warm_relay():
    return {"status": get_hf_auth().warm_relay()}


@router.get("/oauth/start")
def oauth_start():
    result = get_hf_auth().start_oauth()
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result


@router.get("/oauth/status/{session_id}")
def oauth_status(session_id: str):
    return get_hf_auth().oauth_status(session_id)


@router.get("/oauth/callback", response_class=HTMLResponse)
def oauth_callback(code: str | None = None, state: str | None = None,
                   error: str | None = None, error_description: str | None = None):
    auth = get_hf_auth()
    if error:
        if state:
            auth.fail_session(state, error_description or error)
        return HTMLResponse(_result_page(False, error_description or error))
    if not code or not state:
        return HTMLResponse(_result_page(False, "Missing code or state."), status_code=400)
    result = auth.exchange_code(code, state)
    ok = result["status"] == "success"
    message = (f"Signed in as {result.get('username') or 'your account'}." if ok
               else result.get("message", "Authorization failed."))
    return HTMLResponse(_result_page(ok, message))


def _result_page(ok: bool, message: str) -> str:
    """What the login popup shows at the end. Closes itself; the dashboard,
    polling the session, updates on its own."""
    title = "Signed in" if ok else "Sign-in failed"
    color = "#16a34a" if ok else "#dc2626"
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
font:15px/1.45 system-ui,sans-serif;background:#fff;color:#1f2937;text-align:center;padding:16px}}
h1{{color:{color};font-size:22px}}p{{color:#6b7280}}</style></head>
<body><div><h1>{"✓" if ok else "✗"} {title}</h1><p>{html.escape(message)}</p>
<p>You can close this window and go back to the Casquette dashboard.</p></div>
<script>if (window.opener) setTimeout(() => window.close(), 2500);</script></body></html>"""
