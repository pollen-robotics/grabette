"""WiFi status, configuration, and setup page endpoints.

GET  /api/wifi/status      → mode + SSID courant
GET  /api/wifi/credentials → SSID + password du réseau home (subnet hotspot uniquement)
GET  /api/wifi/scan        → liste des réseaux visibles
POST /api/wifi/connect     → connecte grabette au réseau choisi (async, retourne 202)
GET  /api/wifi/setup       → page HTML de configuration (navigateur sur hotspot)
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from grabette.config import settings
from grabette.wifi import (
    get_current_ssid,
    get_local_ip,
    get_network_mode,
    load_home_credentials,
    scan_networks,
    wifi_connect,
)

router = APIRouter(prefix="/api/wifi", tags=["wifi"])

_HOTSPOT_SUBNET = "192.168.42."


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WifiStatus(BaseModel):
    mode: str  # "hotspot" | "connected" | "offline"
    ssid: str | None
    ip: str | None = None


class WifiCredentials(BaseModel):
    ssid: str
    password: str


class ConnectRequest(BaseModel):
    ssid: str
    password: str


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@router.get("/status", response_model=WifiStatus)
def wifi_status() -> WifiStatus:
    return WifiStatus(mode=get_network_mode(), ssid=get_current_ssid(), ip=get_local_ip())


@router.get("/credentials", response_model=WifiCredentials)
def wifi_credentials(request: Request) -> WifiCredentials:
    client_ip = request.client.host if request.client else ""
    if not client_ip.startswith(_HOTSPOT_SUBNET):
        raise HTTPException(
            status_code=403,
            detail="Credentials endpoint is only accessible from the grabette hotspot network",
        )
    creds = load_home_credentials(settings.hotspot_credentials_file)
    if creds is None:
        raise HTTPException(status_code=404, detail="No home network configured yet")
    return WifiCredentials(**creds)


@router.get("/scan")
def wifi_scan() -> list[dict]:
    """Scan and return visible networks sorted by signal strength."""
    return scan_networks()


@router.post("/connect", status_code=202)
def wifi_connect_endpoint(req: ConnectRequest, background_tasks: BackgroundTasks):
    """Connect grabette to the given network. Returns 202 immediately; connection runs in background."""
    background_tasks.add_task(wifi_connect, req.ssid, req.password, settings.hotspot_credentials_file)
    return {"status": "connecting", "ssid": req.ssid}


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
  body { background: #111; color: #eee; font-family: sans-serif; padding: 20px; max-width: 480px; margin: auto; }
  h1 { color: #0ff; font-size: 1.3rem; margin-bottom: 16px; }
  #status { font-size: .85rem; color: #aaa; margin-bottom: 12px; min-height: 1.2em; }
  #status.ok  { color: #0f0; }
  #status.err { color: #f44; }
  #networks { list-style: none; margin-bottom: 16px; }
  #networks li {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 12px; margin-bottom: 4px; border-radius: 6px;
    background: #1e1e1e; cursor: pointer; border: 1px solid #333;
  }
  #networks li:hover { background: #2a2a2a; border-color: #0ff; }
  #networks li.selected { background: #003333; border-color: #0ff; }
  .signal { font-size: .75rem; color: #888; }
  #form { display: none; background: #1e1e1e; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
  #form label { display: block; margin-bottom: 6px; color: #0ff; font-size: .9rem; }
  #form input {
    width: 100%; padding: 8px 10px; border-radius: 4px;
    border: 1px solid #444; background: #111; color: #eee;
    font-size: 1rem; margin-bottom: 12px;
  }
  button {
    padding: 10px 20px; border: none; border-radius: 6px;
    background: #006666; color: #fff; font-size: 1rem; cursor: pointer;
  }
  button:hover { background: #008888; }
  button.secondary { background: #333; margin-left: 8px; }
  #spinner { display: none; color: #0ff; margin-top: 10px; }
</style>
</head>
<body>
<h1>Grabette — WiFi Setup</h1>
<div id="status">Scanning networks…</div>
<ul id="networks"></ul>
<div id="form">
  <label id="net-label">Password for: <strong id="net-name"></strong></label>
  <input type="password" id="password" placeholder="WiFi password" autocomplete="off">
  <button onclick="connect()">Connect</button>
  <button class="secondary" onclick="cancelForm()">Cancel</button>
</div>
<div id="spinner">Connecting, please wait…</div>
<button onclick="scan()" style="margin-top:8px">Refresh networks</button>

<script>
let selectedSsid = null;

async function scan() {
  setStatus('Scanning…');
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
  document.getElementById('form').style.display = 'block';
  document.getElementById('password').focus();
}

function cancelForm() {
  document.getElementById('form').style.display = 'none';
  selectedSsid = null;
}

async function connect() {
  if (!selectedSsid) return;
  const pw = document.getElementById('password').value;
  document.getElementById('form').style.display = 'none';
  document.getElementById('spinner').style.display = 'block';
  setStatus('Connecting to ' + selectedSsid + '…');
  try {
    const r = await fetch('/api/wifi/connect', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ssid: selectedSsid, password: pw})
    });
    if (r.status === 202) {
      setStatus('Connection in progress… checking in 10s', 'ok');
      setTimeout(checkStatus, 10000);
    } else {
      const d = await r.json();
      setStatus('Error: ' + (d.detail || r.status), 'err');
      document.getElementById('spinner').style.display = 'none';
    }
  } catch(e) {
    setStatus('Request failed: ' + e, 'err');
    document.getElementById('spinner').style.display = 'none';
  }
}

async function checkStatus() {
  try {
    const r = await fetch('/api/wifi/status');
    const d = await r.json();
    if (d.mode === 'connected') {
      document.getElementById('spinner').style.display = 'none';
      setStatus('Connected to: ' + d.ssid, 'ok');
    } else {
      setStatus('Still connecting… retrying in 5s');
      setTimeout(checkStatus, 5000);
    }
  } catch(e) {
    setStatus('Grabette unreachable (it may have switched networks). Done!', 'ok');
    document.getElementById('spinner').style.display = 'none';
  }
}

function setStatus(msg, cls) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = cls || '';
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

scan();
</script>
</body>
</html>
"""
