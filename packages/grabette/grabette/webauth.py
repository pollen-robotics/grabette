"""Shared FastAPI login wiring — the HF OAuth/manual-token UI + routes.

    build_auth_router(auth)  -> APIRouter mounted at /api/hf-auth
    LOGIN_CARD               -> HTML+JS snippet for the login card; calls
                                window.grabetteAuthChanged(status) on change
    result_page(ok, msg)     -> the OAuth popup result page
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .auth import HFAuth


class _TokenRequest(BaseModel):
    token: str


def build_auth_router(auth: HFAuth) -> APIRouter:
    router = APIRouter(prefix="/api/hf-auth")

    @router.post("/save-token")
    async def save_token(req: _TokenRequest) -> dict[str, Any]:
        result = auth.save_token(req.token)
        if result["status"] == "error":
            raise HTTPException(400, detail=result.get("message", "Invalid token"))
        return {"status": "success", "username": result.get("username")}

    @router.get("/status")
    async def get_status() -> dict[str, Any]:
        return auth.status()

    @router.delete("/token")
    async def delete_token() -> dict[str, str]:
        if not auth.delete_token():
            raise HTTPException(500, detail="Failed to delete token")
        return {"status": "success"}

    @router.get("/widget", response_class=HTMLResponse)
    async def auth_widget() -> HTMLResponse:
        return HTMLResponse(widget_page())

    @router.get("/oauth/configured")
    async def oauth_configured() -> dict[str, Any]:
        return {"configured": auth.oauth_configured()}

    @router.get("/oauth/start")
    async def oauth_start() -> dict[str, Any]:
        result = auth.start_oauth()
        if result["status"] == "error":
            raise HTTPException(500, detail=result.get("message"))
        return result

    @router.get("/oauth/status/{session_id}")
    async def oauth_status(session_id: str) -> dict[str, Any]:
        return auth.oauth_status(session_id)

    @router.get("/oauth/callback")
    async def oauth_callback(
        request: Request,
        code: str | None = None,
        state: str | None = None,
        error: str | None = None,
        error_description: str | None = None,
    ) -> HTMLResponse:
        if error:
            if state and (s := auth._sessions.get(state)):
                s.status = "error"
                s.error_message = error_description or error
            return HTMLResponse(result_page(False, error_description or error))
        if not code or not state:
            return HTMLResponse(result_page(False, "Missing code or state"), status_code=400)
        result = await auth.exchange_code(code=code, state=state)
        ok = result["status"] == "success"
        msg = (
            f"Successfully logged in as {result.get('username', 'user')}!"
            if ok
            else result.get("message", "Authorization failed")
        )
        return HTMLResponse(result_page(ok, msg))

    return router


def widget_page() -> str:
    """Standalone auth widget served at /api/hf-auth/widget for iframe embedding."""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{margin:0;padding:.8rem;background:transparent;
font-family:-apple-system,system-ui,sans-serif;font-size:.9rem;overflow:hidden}}
.card{{background:#f8fafc;color:#1e293b;padding:1rem;border-radius:10px;border:1px solid #e2e8f0}}
h2{{font-size:.95rem;margin:0 0 .7rem;color:#0f172a}}
input{{box-sizing:border-box;padding:.45rem;border-radius:7px;border:1px solid #cbd5e1;
background:#fff;color:#1e293b;font-family:monospace;width:100%}}
input::placeholder{{color:#94a3b8}}
.row{{display:flex;gap:.5rem;margin-bottom:.5rem}}
.row input{{flex:1}}
button{{padding:.45rem .9rem;border:0;border-radius:7px;cursor:pointer;font-weight:600}}
button.oauth{{background:#10b981;color:#fff;width:100%;margin-bottom:.5rem}}
button.primary{{background:#f59e0b;color:#fff}}
button.logout{{background:#ef4444;color:#fff}}
.muted{{color:#64748b;font-size:.78rem}}
.err{{color:#dc2626;font-size:.78rem;min-height:1rem}}
</style></head>
<body>{LOGIN_CARD}</body></html>"""


def result_page(success: bool, message: str) -> str:
    icon = "✅" if success else "❌"
    title = "Login Successful" if success else "Login Failed"
    color = "#10b981" if success else "#ef4444"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title>
<style>body{{font-family:-apple-system,system-ui,sans-serif;display:flex;justify-content:center;
align-items:center;min-height:100vh;margin:0;background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff}}
.c{{text-align:center;padding:2rem}}.i{{font-size:4rem}}h1{{color:{color}}}p{{color:#a0aec0}}</style></head>
<body><div class="c"><div class="i">{icon}</div><h1>{title}</h1><p>{message}</p>
<p style="color:#667">You can close this window.</p></div>
<script>if(window.opener)setTimeout(()=>window.close(),2500)</script></body></html>"""


# Login card markup + JS. Calls window.grabetteAuthChanged(statusObj) whenever
# auth state changes, so the host page can enable/disable its gated UI.
LOGIN_CARD = """
<div class="card"><h2>HuggingFace login</h2>
 <div id="hfStatus">Checking…</div>
 <div id="hfLogin" style="margin-top:.8rem">
  <button class="oauth" id="hfOauth" style="display:none">One-click login (OAuth)</button>
  <p id="hfOr" class="muted" style="display:none;text-align:center;margin:.5rem 0">or use a token:</p>
  <div class="row"><input id="hfTok" placeholder="hf_..." autocomplete="off">
   <button class="primary" id="hfSave">Save</button></div>
  <div class="err" id="hfErr"></div>
  <p class="muted">Token: <a style="color:#2563eb;text-decoration:underline" href="https://huggingface.co/settings/tokens" target="_blank">hf.co/settings/tokens</a></p>
 </div></div>
<script>
const HF='/api/hf-auth',_$=id=>document.getElementById(id);
async function hfRefresh(){
 const s=await(await fetch(`${HF}/status`)).json();
 if(s.is_logged_in){
  _$('hfStatus').innerHTML=`Logged in as <b>${s.username||'user'}</b> <button class="logout" onclick="hfLogout()">Logout</button>`;
  _$('hfLogin').style.display='none';
 }else{
  _$('hfStatus').textContent='Not logged in.';_$('hfLogin').style.display='block';
  const c=await(await fetch(`${HF}/oauth/configured`)).json();
  const showOauth=c.configured;
  _$('hfOauth').style.display=showOauth?'block':'none';
  _$('hfOr').style.display=showOauth?'block':'none';
 }
 if(window.grabetteAuthChanged)window.grabetteAuthChanged(s);
}
_$('hfSave').onclick=async()=>{_$('hfErr').textContent='';const token=_$('hfTok').value.trim();if(!token)return;
 const r=await fetch(`${HF}/save-token`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});
 if(r.ok){_$('hfTok').value='';hfRefresh();}else{_$('hfErr').textContent=(await r.json()).detail||'failed';}};
async function hfLogout(){await fetch(`${HF}/token`,{method:'DELETE'});hfRefresh();}
_$('hfOauth').onclick=async()=>{const r=await(await fetch(`${HF}/oauth/start`)).json();
 if(r.status!=='success'){_$('hfErr').textContent=r.message;return;}
 const p=window.open(r.auth_url,'hf','width=600,height=750');
 const t=setInterval(async()=>{const st=await(await fetch(`${HF}/oauth/status/${r.session_id}`)).json();
  if(st.status==='authorized'){clearInterval(t);if(p)p.close();hfRefresh();}
  else if(st.status==='error'||st.status==='expired'){clearInterval(t);_$('hfErr').textContent=st.message||'failed';}},1500);};
hfRefresh();
setInterval(hfRefresh,5000);
</script>
"""
