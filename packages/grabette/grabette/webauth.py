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
    async def auth_widget(compact: bool = False, variant: str = "") -> HTMLResponse:
        return HTMLResponse(widget_page(compact=compact, variant=variant))

    @router.get("/oauth/configured")
    async def oauth_configured() -> dict[str, Any]:
        return {"configured": auth.oauth_configured()}

    @router.get("/oauth/warm-relay")
    async def warm_relay() -> dict[str, Any]:
        """Wake the fleet Space so the OAuth callback lands on a running relay
        even with no dashboard tab open. Called by the login card before OAuth."""
        return await auth.warm_relay()

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


# Compact skin for the same card. CSS only, deliberately: the markup and the
# script below are shared with the full widget, so the two can never drift in
# what they actually do — only in how much room they take. Collapses to one
# line when logged in, one row when logged out.
_COMPACT_CSS = """
body{padding:0;font-size:.82rem}
.card{padding:.5rem .65rem;border-radius:9px}
h2{display:none}
/* :not(.hide) matters: both rules carry !important, so specificity decides,
   and an id would otherwise outrank .hide and keep the form on screen once
   logged in. */
#hfLogin:not(.hide){display:flex!important;align-items:center;gap:.5rem;flex-wrap:wrap}
#hfLogin{margin-top:0!important}
button{padding:.35rem .7rem;font-size:.82rem}
button.oauth{width:auto;margin:0;white-space:nowrap}
#hfOr{margin:0!important;color:#94a3b8;font-size:.75rem}
.row{margin:0;flex:1 1 150px}
.row input{padding:.35rem;font-size:.78rem}
/* Only the "get a token here" footnote goes — it is the one thing with nowhere
   to sit in a single row, and the field's placeholder still says what belongs
   in it. #hfOr stays: it is what tells you the two options are alternatives. */
#hfLogin>p.muted:not(#hfOr){display:none}
.err:empty{display:none}
.err{min-height:0}
"""


# The Overview shows this widget as one slab the size of the Fleet button next
# to it: the login has to work from the home page, and a login card there would
# dwarf the thing it sits beside. Same geometry as _ERRAND_BUTTON in ui/app.py —
# two documents, so the values are written twice on purpose.
_BUTTON_CSS = """
body{padding:0;background:transparent;overflow:visible}
.card{background:none;border:none;padding:0}
h2{display:none}
#hfStatus{margin:0}
/* Logged out, the status line is the words "Not logged in." above a button
   that already says so — it only earns its place once it holds the account. */
#hfStatus:not(:has(.who-row)){display:none}
/* Logged out: the OAuth button IS the widget. */
button.oauth{width:100%;height:56px;margin:0;border-radius:9px;
font-size:1rem;font-weight:700;color:#fff;
background:linear-gradient(135deg,#f59e0b,#ef4444);
box-shadow:0 4px 14px rgba(0,0,0,.22)}
button.oauth:disabled{opacity:.7;cursor:default}
/* Logged in: the status line becomes the same slab, in another colour. */
.who-row{height:56px;box-sizing:border-box;padding:0 .75rem 0 1.1rem;
border-radius:9px;color:#fff;font-weight:700;flex-wrap:nowrap;
background:linear-gradient(135deg,#6366f1,#8b5cf6);
box-shadow:0 4px 14px rgba(0,0,0,.22)}
button.logout{background:rgba(255,255,255,.2);color:#fff;
padding:.3rem .75rem;font-size:.8rem;font-weight:600}
/* No gap above: the slab has to start where the Fleet button beside it does. */
#hfLogin{margin-top:0!important}
#hfOr{text-align:center;margin:.5rem 0 0!important;cursor:pointer;
text-decoration:underline}
.err:empty{display:none}
"""

# The token field is a fallback, not the offer: it stays folded away under the
# button until asked for — unless OAuth is unavailable, in which case it is the
# only way in and shows itself.
_BUTTON_JS = """
<script>
(function(){
  var or_=document.getElementById('hfOr'), open=false;
  function extras(){
    return [document.querySelector('#hfLogin .row'),
            document.querySelector('#hfLogin .err')]
      .concat([].slice.call(document.querySelectorAll('#hfLogin > p.muted:not(#hfOr)')));
  }
  function show(on){extras().forEach(function(e){if(e)e.style.display=on?'':'none';});}
  if(or_){
    or_.textContent='or use a token';
    or_.onclick=function(){open=!open;show(open);};
  }
  // hfRefresh hides #hfOr when the device has no OAuth client configured; that
  // is exactly when the token field has to be on screen.
  setInterval(function(){
    show(or_ && or_.classList.contains('hide') ? true : open);
  }, 600);
  show(false);
})();
</script>
"""


def widget_page(compact: bool = False, variant: str = "") -> str:
    """Standalone auth widget served at /api/hf-auth/widget for iframe embedding.

    One card and one script, three skins: the full card for Settings,
    compact=True for a dense status line, and variant="button" for the
    Overview, where the whole widget is a single slab the size of the Fleet
    button. The OAuth flow, the token form and logout are the same code in all
    three — a second login implementation is a second one to keep right.
    """
    extra_css = _BUTTON_CSS if variant == "button" else (_COMPACT_CSS if compact else "")
    extra_js = _BUTTON_JS if variant == "button" else ""
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
.who-row{{display:flex;align-items:center;justify-content:space-between;gap:.8rem;flex-wrap:wrap}}
.muted{{color:#64748b;font-size:.78rem}}
.err{{color:#dc2626;font-size:.78rem;min-height:1rem}}
/* Visibility is a class so a skin can restyle `display` freely — an inline
   display:none set from the script would outrank any stylesheet rule. */
.hide{{display:none!important}}
{extra_css}
</style></head>
<body>{LOGIN_CARD}{extra_js}</body></html>"""


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
  <button class="oauth hide" id="hfOauth">One-click login (OAuth)</button>
  <p id="hfOr" class="muted hide" style="text-align:center;margin:.5rem 0">or use a token:</p>
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
  _$('hfStatus').innerHTML=`<div class="who-row"><span>Logged in as <b>${s.username||'user'}</b></span><button class="logout" onclick="hfLogout()">Logout</button></div>`;
  _$('hfLogin').classList.add('hide');
 }else{
  _$('hfStatus').textContent='Not logged in.';_$('hfLogin').classList.remove('hide');
  const c=await(await fetch(`${HF}/oauth/configured`)).json();
  const showOauth=c.configured;
  _$('hfOauth').classList.toggle('hide',!showOauth);
  _$('hfOr').classList.toggle('hide',!showOauth);
 }
 if(window.grabetteAuthChanged)window.grabetteAuthChanged(s);
}
_$('hfSave').onclick=async()=>{_$('hfErr').textContent='';const token=_$('hfTok').value.trim();if(!token)return;
 const r=await fetch(`${HF}/save-token`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});
 if(r.ok){_$('hfTok').value='';hfRefresh();}else{_$('hfErr').textContent=(await r.json()).detail||'failed';}};
async function hfLogout(){await fetch(`${HF}/token`,{method:'DELETE'});hfRefresh();}
_$('hfOauth').onclick=async()=>{_$('hfErr').textContent='';
 const b=_$('hfOauth'),orig=b.textContent;b.disabled=true;b.textContent='Waking fleet…';
 // Wake the (sleep-when-idle) fleet Space first, so the OAuth callback — routed
 // through it by HF — lands on a running relay even with no dashboard tab open.
 try{await fetch(`${HF}/oauth/warm-relay`);}catch(e){}
 b.disabled=false;b.textContent=orig;
 const r=await(await fetch(`${HF}/oauth/start`)).json();
 if(r.status!=='success'){_$('hfErr').textContent=r.message;return;}
 const p=window.open(r.auth_url,'hf','width=600,height=750');
 const t=setInterval(async()=>{const st=await(await fetch(`${HF}/oauth/status/${r.session_id}`)).json();
  if(st.status==='authorized'){clearInterval(t);if(p)p.close();hfRefresh();}
  else if(st.status==='error'||st.status==='expired'){clearInterval(t);_$('hfErr').textContent=st.message||'failed';}},1500);};
hfRefresh();
setInterval(hfRefresh,5000);
</script>
"""
