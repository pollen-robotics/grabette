"""Multi-device sync orchestrator.

When the local user (typically via the hardware button on a Grabette;
grabette has no button so this is API-driven there) presses start, the
device's `/api/sync/start` endpoint:

  1. Loads the peer list from env var (GRABETTE_PEERS / GRABETTE_PEERS).
  2. Preflight: checks each peer is reachable and NTP-synchronized.
  3. Picks T₀ = now + lead_ms.
  4. Fan-out POST /api/episodes/start { start_at_utc: T₀ } to each peer,
     in parallel.
  5. On any peer failure → rolls back: POSTs /api/episodes/stop to all
     peers that successfully scheduled, refuses local start.
  6. On all peers OK → schedules local start with the same T₀.

`/api/sync/stop` mirrors this for the stop path (no preflight needed).

The asymmetry between `/api/sync/*` (fan-outs) and `/api/episodes/*`
(local-only) keeps the protocol loop-free without an in-band propagate
flag.

Peer config format:
    GRABETTE_PEERS="dev1=http://host1:8001,dev2=http://host2:8000"

(role is left to the future webui / metadata layer; the env-var form is a
placeholder for Phase 2 testing.)
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from grabette.app.dependencies import get_scheduler
from grabette.scheduler import EpisodeScheduler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sync", tags=["sync"])

# Repo-specific env var. Grabette uses GRABETTE_PEERS; casquette uses
# CASQUETTE_PEERS. Same format; only the prefix differs so the two
# daemons don't accidentally pick up each other's config on a shared host.
_PEERS_ENV = "GRABETTE_PEERS"

# How far ahead of "now" the orchestrator schedules T₀. Has to be larger
# than the worst-case fan-out fan-in time (network + handler scheduling)
# so every peer can finish receiving the scheduled-start request before
# T₀ actually fires. 500 ms is comfortably above LAN RTT.
LEAD_MS = 500

# Sync preflight: maximum acceptable |offset| reported by each peer's
# timedatectl. Above this we refuse to schedule. 50 ms is 1.5 frames at
# 30 fps, well below where multi-device alignment would suffer noticeably.
PREFLIGHT_OFFSET_MAX_US = 50_000

# Per-peer HTTP timeouts.
PEER_HTTP_TIMEOUT_S = 2.0


class Peer(BaseModel):
    device_id: str
    url: str  # base URL, e.g. http://host:8001


def _load_peers() -> list[Peer]:
    raw = os.environ.get(_PEERS_ENV, "").strip()
    if not raw:
        return []
    peers: list[Peer] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        device_id, url = chunk.split("=", 1)
        peers.append(Peer(device_id=device_id.strip(), url=url.strip().rstrip("/")))
    return peers


async def _preflight_one(client: httpx.AsyncClient, peer: Peer) -> tuple[Peer, str | None]:
    """Return (peer, None) on OK or (peer, error_message)."""
    try:
        r = await client.get(f"{peer.url}/api/system/time", timeout=PEER_HTTP_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return peer, f"unreachable: {e}"
    if data.get("ntp_synchronized") is False:
        return peer, "ntp not synchronized"
    off = data.get("offset_us")
    if off is not None and abs(off) > PREFLIGHT_OFFSET_MAX_US:
        return peer, f"|offset|={abs(off)} µs > {PREFLIGHT_OFFSET_MAX_US} µs"
    return peer, None


async def _peer_start(
    client: httpx.AsyncClient, peer: Peer, target_iso: str,
) -> tuple[Peer, dict | None, str | None]:
    """Returns (peer, body_on_success, error_on_failure)."""
    try:
        r = await client.post(
            f"{peer.url}/api/episodes/start",
            json={"start_at_utc": target_iso},
            timeout=PEER_HTTP_TIMEOUT_S,
        )
    except Exception as e:
        return peer, None, f"{type(e).__name__}: {e}"
    if r.status_code != 200:
        return peer, None, f"HTTP {r.status_code}: {r.text[:200]}"
    return peer, r.json(), None


async def _peer_stop(
    client: httpx.AsyncClient, peer: Peer,
) -> tuple[Peer, str | None]:
    try:
        r = await client.post(
            f"{peer.url}/api/episodes/stop",
            timeout=PEER_HTTP_TIMEOUT_S,
        )
    except Exception as e:
        return peer, f"{type(e).__name__}: {e}"
    if r.status_code not in (200, 409):
        return peer, f"HTTP {r.status_code}: {r.text[:200]}"
    return peer, None


class SyncStartResponse(BaseModel):
    scheduled_start_utc: str
    local_episode_id: str
    peers: list[dict]


@router.post("/start", response_model=SyncStartResponse)
async def sync_start(scheduler: EpisodeScheduler = Depends(get_scheduler)):
    peers = _load_peers()
    target = datetime.now(timezone.utc) + timedelta(milliseconds=LEAD_MS)
    target_iso = target.isoformat()

    async with httpx.AsyncClient() as client:
        # ── Preflight (skipped if no peers) ──────────────────────────
        if peers:
            results = await asyncio.gather(
                *(_preflight_one(client, p) for p in peers)
            )
            preflight_fails = [(p, e) for p, e in results if e is not None]
            if preflight_fails:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "phase": "preflight",
                        "message": "one or more peers failed preflight",
                        "failures": [
                            {"peer": p.device_id, "url": p.url, "error": e}
                            for p, e in preflight_fails
                        ],
                    },
                )

        # ── Fan-out start to all peers ──────────────────────────────
        peer_results: list[tuple[Peer, dict | None, str | None]] = []
        if peers:
            peer_results = list(await asyncio.gather(
                *(_peer_start(client, p, target_iso) for p in peers)
            ))

        successes = [(p, body) for p, body, err in peer_results if err is None]
        failures = [(p, err) for p, _b, err in peer_results if err is not None]

        if failures:
            # Roll back successful peers, then refuse.
            if successes:
                await asyncio.gather(
                    *(_peer_stop(client, p) for p, _ in successes),
                    return_exceptions=True,
                )
            raise HTTPException(
                status_code=502,
                detail={
                    "phase": "peer_start",
                    "message": "peer(s) failed; rolled back",
                    "failures": [
                        {"peer": p.device_id, "url": p.url, "error": e}
                        for p, e in failures
                    ],
                    "rolled_back": [p.device_id for p, _ in successes],
                },
            )

        # ── Local schedule (after peers are committed) ──────────────
        try:
            local_episode_id = await scheduler.start(start_at_utc=target)
        except (RuntimeError, ValueError) as e:
            # Local refused → undo everything we just told peers to do.
            if successes:
                await asyncio.gather(
                    *(_peer_stop(client, p) for p, _ in successes),
                    return_exceptions=True,
                )
            raise HTTPException(
                status_code=409,
                detail={"phase": "local_start", "error": str(e)},
            )

    return SyncStartResponse(
        scheduled_start_utc=target_iso,
        local_episode_id=local_episode_id,
        peers=[
            {"device_id": p.device_id, "url": p.url,
             "episode_id": body.get("episode_id") if body else None}
            for p, body in successes
        ],
    )


class SyncStopResponse(BaseModel):
    local: dict | None
    peers: list[dict]


@router.post("/stop", response_model=SyncStopResponse)
async def sync_stop(scheduler: EpisodeScheduler = Depends(get_scheduler)):
    peers = _load_peers()

    # Stop local first (best-effort) — if it was already idle, that's fine.
    local_result: dict | None = None
    try:
        status = await scheduler.stop()
        if status is None:
            local_result = {"status": "cancelled"}
        else:
            # backend status models are pydantic objects; .model_dump if available
            local_result = (
                status.model_dump() if hasattr(status, "model_dump") else dict(status)
            )
    except RuntimeError:
        # Was IDLE locally — that's fine for stop semantics.
        local_result = {"status": "idle"}

    # Fan-out stop to peers; don't fail the whole call on individual errors.
    peer_results: list[dict] = []
    if peers:
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(
                *(_peer_stop(client, p) for p in peers),
                return_exceptions=True,
            )
        for peer, result in zip(peers, results):
            if isinstance(result, Exception):
                peer_results.append({"device_id": peer.device_id,
                                     "error": f"{type(result).__name__}: {result}"})
                continue
            _p, err = result
            peer_results.append({
                "device_id": peer.device_id,
                "url": peer.url,
                **({"error": err} if err else {"status": "ok"}),
            })

    return SyncStopResponse(local=local_result, peers=peer_results)


@router.get("/peers")
def list_peers():
    """Echo the configured peer list. Useful as a config sanity check."""
    return {
        "env_var": _PEERS_ENV,
        "peers": [p.model_dump() for p in _load_peers()],
    }
