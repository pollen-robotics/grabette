"""WiFi status, configuration, and setup page endpoints.

GET  /api/wifi/status      → mode + current SSID
GET  /api/wifi/scan        → list of visible networks that are NOT already known
GET  /api/wifi/saved       → list of networks this device already has credentials for
POST /api/wifi/connect     → connects grabette to the chosen network (async, returns 202)
POST /api/wifi/switch      → switches to an already-known network (async, returns 202)
GET  /api/wifi/connect-result → result of the last connection attempt
GET  /api/wifi/setup       → HTML network panel embedded in the dashboard's Home page
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from grabette.wifi import (
    activate_saved_network,
    get_current_ssid,
    get_local_ip,
    get_network_mode,
    list_saved_networks,
    scan_networks,
    wifi_connect,
)

router = APIRouter(prefix="/api/wifi", tags=["wifi"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WifiStatus(BaseModel):
    mode: str  # "hotspot" | "connected" | "offline"
    ssid: str | None
    ip: str | None = None


class ConnectRequest(BaseModel):
    ssid: str
    password: str


class SwitchRequest(BaseModel):
    ssid: str


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.get("/status", response_model=WifiStatus)
def wifi_status() -> WifiStatus:
    return WifiStatus(mode=get_network_mode(), ssid=get_current_ssid(), ip=get_local_ip())


@router.get("/scan")
def wifi_scan() -> list[dict]:
    """Scan and return visible networks sorted by signal strength."""
    return scan_networks()


@router.get("/saved")
def wifi_saved() -> list[dict]:
    """Networks this device already has credentials for — switchable in one click."""
    return list_saved_networks()


# Result of the last connection attempt — read by /api/wifi/connect-result
_last_connect: dict = {"status": "idle", "message": ""}


def _do_connect(ssid: str, password: str) -> None:
    global _last_connect
    _last_connect = {"status": "connecting", "message": f"Connecting to {ssid}…"}
    logger.info("[wifi] _do_connect started: ssid=%s", ssid)
    try:
        result = wifi_connect(ssid, password)
        logger.info("[wifi] wifi_connect result: %s", result)
        if result.startswith("OK:"):
            _last_connect = {"status": "ok", "message": result}
        else:
            _last_connect = {"status": "error", "message": result}
    except Exception as exc:
        logger.exception("[wifi] _do_connect exception: %s", exc)
        _last_connect = {"status": "error", "message": f"ERROR: {exc}"}


@router.post("/connect", status_code=202)
def wifi_connect_endpoint(req: ConnectRequest, background_tasks: BackgroundTasks):
    """Connect grabette to the given network. Returns 202 immediately; connection runs in background."""
    global _last_connect
    _last_connect = {"status": "connecting", "message": f"Connecting to {req.ssid}…"}
    background_tasks.add_task(_do_connect, req.ssid, req.password)
    return {"status": "connecting", "ssid": req.ssid}


def _do_switch(ssid: str) -> None:
    global _last_connect
    _last_connect = {"status": "connecting", "message": f"Switching to {ssid}…"}
    logger.info("[wifi] _do_switch started: ssid=%s", ssid)
    try:
        result = activate_saved_network(ssid)
        logger.info("[wifi] activate_saved_network result: %s", result)
        if result.startswith("OK:"):
            _last_connect = {"status": "ok", "message": result}
        else:
            _last_connect = {"status": "error", "message": result}
    except Exception as exc:
        logger.exception("[wifi] _do_switch exception: %s", exc)
        _last_connect = {"status": "error", "message": f"ERROR: {exc}"}


@router.post("/switch", status_code=202)
def wifi_switch_endpoint(req: SwitchRequest, background_tasks: BackgroundTasks):
    """Switch to an already-known network — no password needed. Returns 202 immediately.

    Shares ``_last_connect`` with /connect so the page can poll one endpoint for
    the outcome of either kind of network change.
    """
    global _last_connect
    _last_connect = {"status": "connecting", "message": f"Switching to {req.ssid}…"}
    background_tasks.add_task(_do_switch, req.ssid)
    return {"status": "connecting", "ssid": req.ssid}


@router.get("/connect-result")
def wifi_connect_result() -> dict:
    """Return the result of the last connection attempt."""
    return _last_connect


# ---------------------------------------------------------------------------
# Web setup page
# ---------------------------------------------------------------------------

@router.get("/setup", response_class=HTMLResponse)
def wifi_setup_page() -> str:
    return _WIFI_SETUP_HTML


_WIFI_SETUP_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grabette — Network</title>
<style>
  /* Visual language mirrors the grabette-fleet dashboard and the BT tool: the
     same translucent cards, muted #a0aec0 labels and emerald→blue primary.
     The body itself is TRANSPARENT — this page is iframed into the dashboard's
     Home page, which already paints the fleet gradient behind it, so an opaque
     background here would read as a box floating on top of the page. */
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: transparent; color: #fff;
    font-family: -apple-system, system-ui, sans-serif;
    padding: 2px; max-width: 620px;
  }
  h2 {
    font-size: .78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .07em; color: #8b98ad; margin: 1.1rem 0 .5rem;
  }
  h2:first-of-type { margin-top: 0; }
  #status { font-size: .82rem; color: #a0aec0; margin-bottom: .6rem; min-height: 1.2em; }
  #status.ok  { color: #6ee7b7; }
  #status.err { color: #fca5a5; }
  ul { list-style: none; }
  li {
    display: flex; justify-content: space-between; align-items: center; gap: .6rem;
    padding: .6rem .75rem; margin-bottom: .35rem; border-radius: 10px;
    background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.1);
    font-size: .88rem;
  }
  #networks li { cursor: pointer; }
  #networks li:hover, #networks li.selected {
    background: rgba(255,255,255,.12); border-color: rgba(59,130,246,.55);
  }
  .ssid { overflow-wrap: anywhere; }
  .signal { font-size: .72rem; color: #8b98ad; white-space: nowrap; }
  /* The network you are on is not a switch target — it is state. */
  li.current { border-color: rgba(16,185,129,.45); background: rgba(16,185,129,.1); }
  .tag {
    font-size: .68rem; font-weight: 700; padding: .08rem .5rem; border-radius: 999px;
    white-space: nowrap; background: rgba(16,185,129,.22); color: #a7f3d0;
  }
  .empty { color: #8b98ad; font-size: .82rem; padding: .2rem 0 .4rem; }
  #form {
    display: none; background: rgba(255,255,255,.06); border-radius: 12px;
    padding: .9rem; margin-bottom: .7rem; border: 1px solid rgba(255,255,255,.12);
  }
  #form label { display: block; margin-bottom: .45rem; color: #c3cbe0; font-size: .85rem; }
  .pw-row { display: flex; gap: .5rem; margin-bottom: .7rem; }
  .pw-row input {
    flex: 1; padding: .5rem .6rem; border-radius: 8px; font-size: .95rem;
    border: 1px solid rgba(255,255,255,.22); background: #16213e; color: #fff;
  }
  .pw-row input:focus { outline: none; border-color: rgba(59,130,246,.7); }
  #error-box {
    display: none; background: rgba(239,68,68,.14); border: 1px solid rgba(248,113,113,.4);
    border-radius: 10px; padding: .6rem .75rem; margin-bottom: .7rem;
    font-size: .82rem; color: #fca5a5; word-break: break-word;
  }
  button {
    padding: .45rem .9rem; border: 0; border-radius: 8px; cursor: pointer;
    font-weight: 600; font-size: .85rem; font-family: inherit;
    background: linear-gradient(135deg,#10b981,#3b82f6); color: #fff;
  }
  button:hover { filter: brightness(1.1); }
  button:active { transform: scale(.97); }
  button:disabled { opacity: .45; cursor: not-allowed; filter: none; }
  button.ghost {
    background: rgba(255,255,255,.08); color: #c3cbe0;
    border: 1px solid rgba(255,255,255,.18);
  }
  button.ghost:hover { background: rgba(255,255,255,.14); filter: none; }
  button.small { padding: .3rem .7rem; font-size: .78rem; }
  #spinner { display: none; color: #c3cbe0; font-size: .85rem; margin-bottom: .6rem; }
</style>
</head>
<body>

<h2>Known networks</h2>
<div id="known-empty" class="empty">Looking for saved networks…</div>
<ul id="known"></ul>

<h2>Other networks</h2>
<div id="status">Scanning networks…</div>
<div id="error-box"></div>
<div id="spinner">Connecting, please wait…</div>
<ul id="networks"></ul>
<div id="form">
  <label id="net-label">Password for: <strong id="net-name"></strong></label>
  <div class="pw-row">
    <input type="password" id="password" placeholder="WiFi password" autocomplete="off"
           onkeydown="if(event.key==='Enter') connect()">
    <button type="button" class="ghost" id="pw-toggle" onclick="togglePw()">Show</button>
  </div>
  <button onclick="connect()">Connect</button>
  <button class="ghost" onclick="cancelForm()">Cancel</button>
</div>
<button class="ghost small" onclick="refresh()">Refresh</button>

<script>
let selectedSsid = null;
let checkAttempts = 0;
const MAX_CHECKS = 30; // 30 × 3 s = 90 s max

function fetchWithTimeout(url, timeout = 5000) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), timeout);
  return fetch(url, { signal: ctrl.signal }).finally(() => clearTimeout(id));
}

// ── Known networks: one click, no password ────────────────────────────────
async function loadKnown() {
  const ul = document.getElementById('known');
  const empty = document.getElementById('known-empty');
  try {
    const r = await fetch('/api/wifi/saved');
    const nets = await r.json();
    ul.innerHTML = '';
    if (!nets.length) {
      empty.textContent = 'No saved networks yet — pick one below to add the first.';
      empty.style.display = 'block';
      return;
    }
    empty.style.display = 'none';
    nets.forEach(n => {
      const li = document.createElement('li');
      if (n.active) li.className = 'current';
      const name = document.createElement('span');
      name.className = 'ssid';
      name.textContent = n.ssid;
      li.appendChild(name);
      if (n.active) {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.textContent = 'Connected';
        li.appendChild(tag);
      } else {
        const btn = document.createElement('button');
        btn.className = 'small';
        btn.textContent = 'Switch';
        btn.onclick = () => switchTo(n.ssid, btn);
        li.appendChild(btn);
      }
      ul.appendChild(li);
    });
  } catch (e) {
    empty.textContent = 'Could not read saved networks: ' + e;
    empty.style.display = 'block';
  }
}

async function switchTo(ssid, btn) {
  hideError();
  cancelForm();
  document.querySelectorAll('#known button').forEach(b => b.disabled = true);
  btn.textContent = 'Switching…';
  document.getElementById('spinner').style.display = 'block';
  setStatus('Switching to ' + ssid + '…');
  checkAttempts = 0;
  try {
    const r = await fetch('/api/wifi/switch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ssid: ssid})
    });
    if (r.status === 202) {
      setTimeout(checkStatus, 3000);
    } else {
      const d = await r.json();
      showError('HTTP ' + r.status + ': ' + (d.detail || 'Unknown error'));
      document.getElementById('spinner').style.display = 'none';
      loadKnown();
    }
  } catch (e) {
    // Losing the device mid-switch is the expected outcome, not a failure:
    // it means grabette left the network this page is served over.
    document.getElementById('spinner').style.display = 'none';
    setStatus('✓ Grabette switched to ' + ssid + '.', 'ok');
  }
}

// ── Other networks: scan + password ───────────────────────────────────────
async function scan() {
  setStatus('Scanning…');
  hideError();
  document.getElementById('networks').innerHTML = '';
  try {
    const r = await fetch('/api/wifi/scan');
    const nets = await r.json();
    if (!nets.length) { setStatus('No other networks in range.'); return; }
    setStatus('Select a network to add it:');
    const ul = document.getElementById('networks');
    nets.forEach(n => {
      const li = document.createElement('li');
      const name = document.createElement('span');
      name.className = 'ssid';
      name.textContent = n.ssid;
      const sig = document.createElement('span');
      sig.className = 'signal';
      sig.textContent = n.signal + '%';
      li.appendChild(name);
      li.appendChild(sig);
      li.onclick = () => selectNet(n.ssid, li);
      ul.appendChild(li);
    });
  } catch(e) { setStatus('Scan failed: ' + e, 'err'); }
}

function refresh() {
  loadKnown();
  scan();
}

function selectNet(ssid, el) {
  document.querySelectorAll('#networks li').forEach(l => l.classList.remove('selected'));
  el.classList.add('selected');
  selectedSsid = ssid;
  document.getElementById('net-name').textContent = ssid;
  document.getElementById('password').value = '';
  document.getElementById('pw-toggle').textContent = 'Show';
  document.getElementById('password').type = 'password';
  hideError();
  document.getElementById('form').style.display = 'block';
  document.getElementById('password').focus();
}

function cancelForm() {
  document.getElementById('form').style.display = 'none';
  selectedSsid = null;
  hideError();
}

function togglePw() {
  const pw = document.getElementById('password');
  const btn = document.getElementById('pw-toggle');
  if (pw.type === 'password') { pw.type = 'text';     btn.textContent = 'Hide'; }
  else                        { pw.type = 'password'; btn.textContent = 'Show'; }
}

async function connect() {
  if (!selectedSsid) return;
  const pw = document.getElementById('password').value;
  hideError();
  document.getElementById('form').style.display = 'none';
  document.getElementById('spinner').style.display = 'block';
  setStatus('Connecting to ' + selectedSsid + '…');
  checkAttempts = 0;
  try {
    const r = await fetch('/api/wifi/connect', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ssid: selectedSsid, password: pw})
    });
    if (r.status === 202) {
      setTimeout(checkStatus, 3000);
    } else {
      const d = await r.json();
      showError('HTTP ' + r.status + ': ' + (d.detail || 'Unknown error'));
      document.getElementById('spinner').style.display = 'none';
      document.getElementById('form').style.display = 'block';
    }
  } catch(e) {
    showError('Request failed: ' + e);
    document.getElementById('spinner').style.display = 'none';
    document.getElementById('form').style.display = 'block';
  }
}

async function checkStatus() {
  checkAttempts++;
  try {
    const [wifiRes, connRes] = await Promise.all([
      fetchWithTimeout('/api/wifi/status'),
      fetchWithTimeout('/api/wifi/connect-result')
    ]);
    const wifi = await wifiRes.json();
    const conn = await connRes.json();

    if (conn.status === 'error') {
      document.getElementById('spinner').style.display = 'none';
      showError(conn.message);
      setStatus('Connection failed.', 'err');
      if (selectedSsid) document.getElementById('form').style.display = 'block';
      loadKnown();
      return;
    }

    if (wifi.mode === 'connected' && conn.status === 'ok') {
      document.getElementById('spinner').style.display = 'none';
      setStatus('✓ Connected to: ' + wifi.ssid, 'ok');
      refresh();
      return;
    }

    if (checkAttempts >= MAX_CHECKS) {
      document.getElementById('spinner').style.display = 'none';
      showError('Connection timed out. Check the password and try again.');
      setStatus('Connection timed out.', 'err');
      if (selectedSsid) document.getElementById('form').style.display = 'block';
      loadKnown();
      return;
    }

    setStatus('Connecting… (' + checkAttempts + ')');
    setTimeout(checkStatus, 3000);
  } catch(e) {
    // Grabette unreachable = it switched networks = success
    document.getElementById('spinner').style.display = 'none';
    setStatus('✓ Grabette switched to the new network.', 'ok');
  }
}

function setStatus(msg, cls) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = cls || '';
}

function showError(msg) {
  const el = document.getElementById('error-box');
  el.textContent = msg;
  el.style.display = 'block';
}

function hideError() {
  document.getElementById('error-box').style.display = 'none';
}

refresh();
</script>
</body>
</html>
"""
