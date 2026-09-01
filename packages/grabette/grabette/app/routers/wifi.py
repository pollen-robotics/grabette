"""WiFi status, configuration, and setup page endpoints.

GET  /api/wifi/status      → mode + current SSID
GET  /api/wifi/scan        → list of visible networks that are NOT already known
GET  /api/wifi/saved       → list of networks this device already has credentials for
POST /api/wifi/connect     → joins a new network by SSID + password (async, returns 202)
POST /api/wifi/switch      → switches to an already-known network (async, returns 202)
GET  /api/wifi/connect-result → result of the last connection attempt
GET  /api/wifi/setup       → the "Switch network" panel iframed by the Home page

The setup panel only switches between KNOWN networks. Joining a new one is the
Bluetooth tool's job (docs/index.html → bluetooth/bluetooth_service.py), because
that path works when the device is on no network the operator can reach — which
is exactly when it is needed, and when this page cannot be loaded at all.
/scan and /connect stay: they are the same capability over HTTP, still reachable
over the device's own hotspot, and grabette-fleet builds on this API surface.
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
<title>Grabette — Known networks</title>
<style>
  /* Same palette as the dashboard that iframes this page. It is a separate
     document, so it cannot inherit the parent's body.dark — it reads the same
     localStorage key on load (same origin), and the parent pushes the class on
     it when the theme is toggled while it is already open. */
  :root {
    --gb-card: #ffffff;
    --gb-sunk: rgba(22,33,62,.05);
    --gb-border: rgba(22,33,62,.14);
    --gb-text: #16213e;
    --gb-soft: #3f4a5f;
    --gb-muted: #5f6b80;
    --gb-ok: #047857;
    --gb-ok-bg: rgba(16,185,129,.12);
    --gb-ok-bd: rgba(4,120,87,.45);
    --gb-bad: #b91c1c;
    --gb-bad-bg: rgba(239,68,68,.07);
    --gb-bad-bd: rgba(185,28,28,.4);
  }
  body.dark {
    --gb-card: rgba(255,255,255,.06);
    --gb-sunk: rgba(255,255,255,.08);
    --gb-border: rgba(255,255,255,.12);
    --gb-text: #ffffff;
    --gb-soft: #c3cbe0;
    --gb-muted: #8b98ad;
    --gb-ok: #6ee7b7;
    --gb-ok-bg: rgba(16,185,129,.1);
    --gb-ok-bd: rgba(16,185,129,.45);
    --gb-bad: #fca5a5;
    --gb-bad-bg: rgba(239,68,68,.14);
    --gb-bad-bd: rgba(248,113,113,.4);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  /* Transparent: the dashboard already paints its gradient behind this. */
  body {
    background: transparent; color: var(--gb-text);
    font-family: -apple-system, system-ui, sans-serif;
    padding: 2px;
  }
  #status { font-size: .82rem; color: var(--gb-muted); margin-bottom: .6rem; min-height: 1.2em; }
  #status.ok  { color: var(--gb-ok); }
  #status.err { color: var(--gb-bad); }
  ul { list-style: none; }
  li {
    display: flex; justify-content: space-between; align-items: center; gap: .6rem;
    padding: .6rem .75rem; margin-bottom: .35rem; border-radius: 10px;
    background: var(--gb-card); border: 1px solid var(--gb-border);
    font-size: .88rem;
  }
  .ssid { overflow-wrap: anywhere; font-weight: 600; }
  /* The network you are on is not a switch target — it is state. */
  li.current { border-color: var(--gb-ok-bd); background: var(--gb-ok-bg); }
  .tag {
    font-size: .68rem; font-weight: 700; padding: .1rem .55rem; border-radius: 999px;
    white-space: nowrap; background: var(--gb-ok-bg); color: var(--gb-ok);
    border: 1px solid var(--gb-ok-bd);
  }
  .empty { color: var(--gb-muted); font-size: .82rem; line-height: 1.5; padding: .2rem 0 .5rem; }
  #error-box {
    display: none; background: var(--gb-bad-bg); border: 1px solid var(--gb-bad-bd);
    border-radius: 10px; padding: .6rem .75rem; margin-bottom: .7rem;
    font-size: .82rem; color: var(--gb-bad); word-break: break-word;
  }
  button {
    padding: .42rem .85rem; border: 0; border-radius: 8px; cursor: pointer;
    font-weight: 600; font-size: .82rem; font-family: inherit;
    background: linear-gradient(135deg,#10b981,#3b82f6); color: #fff;
    white-space: nowrap;
  }
  button:hover { filter: brightness(1.08); }
  button:active { transform: scale(.97); }
  button:disabled { opacity: .45; cursor: not-allowed; filter: none; }
  button.ghost {
    background: var(--gb-sunk); color: var(--gb-soft);
    border: 1px solid var(--gb-border);
  }
  button.ghost:hover { filter: none; background: var(--gb-border); }
  #spinner { display: none; color: var(--gb-soft); font-size: .85rem; margin-bottom: .6rem; }
  @media (max-width: 480px) {
    li { flex-wrap: wrap; }
    li button, li .tag { margin-left: auto; }
  }
</style>
</head>
<body>

<div id="status"></div>
<div id="error-box"></div>
<div id="spinner">Switching, please wait…</div>
<div id="known-empty" class="empty">Looking for saved networks…</div>
<ul id="known"></ul>
<button class="ghost" onclick="loadKnown()">Refresh</button>

<script>
// Theme: same key the dashboard writes, read before first paint.
(function () {
  try {
    var t = localStorage.getItem('grabette-theme')
      || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    if (t === 'dark') { document.body.classList.add('dark'); }
  } catch (e) {}
})();

let checkAttempts = 0;
const MAX_CHECKS = 30; // 30 × 3 s = 90 s max

function fetchWithTimeout(url, timeout = 5000) {
  const ctrl = new AbortController();
  const id = setTimeout(() => ctrl.abort(), timeout);
  return fetch(url, { signal: ctrl.signal }).finally(() => clearTimeout(id));
}

// This panel only ever SWITCHES between networks grabette already has
// credentials for. Joining a network it has never seen is the Bluetooth tool's
// job — it works when the device is unreachable, which is when it is needed.
async function loadKnown() {
  const ul = document.getElementById('known');
  const empty = document.getElementById('known-empty');
  try {
    const r = await fetch('/api/wifi/saved');
    const nets = await r.json();
    ul.innerHTML = '';
    if (!nets.length) {
      empty.textContent = 'No saved networks yet. Use the Bluetooth tool below '
        + 'to join the first one — after that it shows up here.';
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
      setStatus('Switch failed.', 'err');
      loadKnown();
      return;
    }

    if (wifi.mode === 'connected' && conn.status === 'ok') {
      document.getElementById('spinner').style.display = 'none';
      setStatus('✓ Connected to: ' + wifi.ssid, 'ok');
      loadKnown();
      return;
    }

    if (checkAttempts >= MAX_CHECKS) {
      document.getElementById('spinner').style.display = 'none';
      showError('Timed out. The network may be out of range.');
      setStatus('Switch timed out.', 'err');
      loadKnown();
      return;
    }

    setStatus('Switching… (' + checkAttempts + ')');
    setTimeout(checkStatus, 3000);
  } catch (e) {
    // Grabette unreachable = it switched networks = success
    document.getElementById('spinner').style.display = 'none';
    setStatus('✓ Grabette switched to the new network.', 'ok');
  }
}

function setStatus(msg, cls) {
  const el = document.getElementById('status');
  el.textContent = msg || '';
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

loadKnown();
</script>
</body>
</html>
"""
