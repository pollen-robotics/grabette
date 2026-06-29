"""WiFi status, configuration, and setup page endpoints.

GET  /api/wifi/status      → mode + current SSID
GET  /api/wifi/scan        → list of visible networks
POST /api/wifi/connect     → connects grabette to the chosen network (async, returns 202)
GET  /api/wifi/connect-result → result of the last connection attempt
GET  /api/wifi/setup       → HTML configuration page embedded in the Settings page
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from grabette.wifi import (
    get_current_ssid,
    get_local_ip,
    get_network_mode,
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
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grabette — WiFi Setup</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #ffffff; color: #111827; font-family: sans-serif; padding: 20px; max-width: 480px; margin: auto; }
  h1 { color: #f97316; font-size: 1.3rem; margin-bottom: 16px; }
  #status { font-size: .85rem; color: #64748b; margin-bottom: 12px; min-height: 1.2em; }
  #status.ok  { color: #16a34a; }
  #status.err { color: #dc2626; }
  #networks { list-style: none; margin-bottom: 16px; }
  #networks li {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 12px; margin-bottom: 4px; border-radius: 6px;
    background: #f8fafc; cursor: pointer; border: 1px solid #e2e8f0;
  }
  #networks li:hover { background: #fff7ed; border-color: #f97316; }
  #networks li.selected { background: #fff7ed; border-color: #f97316; }
  .signal { font-size: .75rem; color: #94a3b8; }
  #form { display: none; background: #f8fafc; border-radius: 8px; padding: 16px; margin-bottom: 12px; border: 1px solid #e2e8f0; }
  #form label { display: block; margin-bottom: 6px; color: #f97316; font-size: .9rem; }
  .pw-row { display: flex; gap: 8px; margin-bottom: 12px; }
  .pw-row input {
    flex: 1; padding: 8px 10px; border-radius: 4px;
    border: 1px solid #cbd5e1; background: #ffffff; color: #111827; font-size: 1rem;
  }
  .pw-row .toggle {
    padding: 8px 14px; background: #f1f5f9; border: 1px solid #cbd5e1;
    border-radius: 4px; color: #475569; font-size: .85rem; cursor: pointer; white-space: nowrap;
  }
  .pw-row .toggle:hover { background: #e2e8f0; }
  #error-box {
    display: none; background: #fef2f2; border: 1px solid #fca5a5; border-radius: 6px;
    padding: 10px 14px; margin-bottom: 12px; font-size: .85rem; color: #dc2626;
    word-break: break-word;
  }
  button {
    padding: 10px 20px; border: none; border-radius: 6px;
    background: #f97316; color: #fff; font-size: 1rem; cursor: pointer; font-weight: 600;
  }
  button:hover { background: #ea6c0a; }
  button:active { background: #c2410c; transform: scale(0.97); }
  button.secondary { background: #f1f5f9; color: #374151; font-weight: 400; margin-left: 8px; border: 1px solid #e2e8f0; }
  button.secondary:hover { background: #e2e8f0; }
  #spinner { display: none; color: #f97316; margin-top: 10px; }
</style>
</head>
<body>
<h1>Grabette — WiFi Setup</h1>
<div id="status">Scanning networks…</div>
<div id="error-box"></div>
<ul id="networks"></ul>
<div id="form">
  <label id="net-label">Password for: <strong id="net-name"></strong></label>
  <div class="pw-row">
    <input type="password" id="password" placeholder="WiFi password" autocomplete="off"
           onkeydown="if(event.key==='Enter') connect()">
    <button type="button" class="toggle" id="pw-toggle" onclick="togglePw()">Show</button>
  </div>
  <button onclick="connect()">Connect</button>
  <button class="secondary" onclick="cancelForm()">Cancel</button>
</div>
<div id="spinner">Connecting, please wait…</div>
<button onclick="scan()" style="margin-top:8px">Refresh networks</button>

<script>
let selectedSsid = null;
let checkAttempts = 0;
const MAX_CHECKS = 30; // 30 × 3 s = 90 s max

async function scan() {
  setStatus('Scanning…');
  hideError();
  document.getElementById('networks').innerHTML = '';
  try {
    const r = await fetch('/api/wifi/scan');
    const nets = await r.json();
    if (!nets.length) { setStatus('No networks found.', 'err'); return; }
    setStatus('Select a network:');
    const ul = document.getElementById('networks');
    nets.forEach(n => {
      const li = document.createElement('li');
      li.innerHTML = '<span>' + escHtml(n.ssid) + '</span><span class="signal">' + n.signal + '%</span>';
      li.onclick = () => selectNet(n.ssid, li);
      ul.appendChild(li);
    });
  } catch(e) { setStatus('Scan failed: ' + e, 'err'); }
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
  setStatus('Connecting to ' + escHtml(selectedSsid) + '…');
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
      fetch('/api/wifi/status'),
      fetch('/api/wifi/connect-result')
    ]);
    const wifi = await wifiRes.json();
    const conn = await connRes.json();

    if (conn.status === 'error') {
      document.getElementById('spinner').style.display = 'none';
      showError(conn.message);
      setStatus('Connection failed.', 'err');
      document.getElementById('form').style.display = 'block';
      return;
    }

    if (wifi.mode === 'connected') {
      document.getElementById('spinner').style.display = 'none';
      setStatus('✓ Connected to: ' + wifi.ssid, 'ok');
      return;
    }

    if (checkAttempts >= MAX_CHECKS) {
      document.getElementById('spinner').style.display = 'none';
      showError('Connection timed out. Check the password and try again.');
      setStatus('Connection timed out.', 'err');
      document.getElementById('form').style.display = 'block';
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

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

scan();
</script>
</body>
</html>
"""
