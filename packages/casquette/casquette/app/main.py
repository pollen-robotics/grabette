from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time as _time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from casquette.config import settings
from casquette.daemon import Daemon, DaemonState

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_daemon: Daemon | None = None

# A synchronized (group) start delivered more than this many seconds past its T0
# is too late to be usable as multi-device data — refuse rather than keep a
# desynced episode. (Mirrors grabette's MAX_START_LATENESS_S.)
MAX_START_LATENESS_S = 1.0


def _exc_text(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


# Episodes per Hub commit (the Hub caps commits at 320/hour per repo).
_UPLOAD_BATCH_EPISODES = 10


def _is_rate_limited(e: BaseException) -> bool:
    """A 429 anywhere in the exception chain (duck-typed: no hub import here)."""
    seen: set[int] = set()   # guards against cyclic chains
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if getattr(getattr(e, "response", None), "status_code", None) == 429:
            return True
        e = e.__cause__ or e.__context__
    return False


def _read_hf_token() -> str | None:
    """Read the HF token WITHOUT importing huggingface_hub.

    The relay only needs the token string, but importing huggingface_hub costs
    ~11 MB resident + a slow cold-boot import — and on the 512 MB Pi Zero 2W that
    marginal memory is what tipped the device into swap-thrash and wedged the CSI
    camera (frontend timeout). Mirror huggingface_hub.get_token()'s resolution:
    HF_TOKEN env var, else the token file at {HF_HOME or ~/.cache/huggingface}/token
    (honoring HF_TOKEN_PATH). The full library is still used lazily for the actual
    upload (fleet/hf.py) — just not at startup. Read on every call, so a token
    renewed by the OAuth refresh loop is picked up at once.
    """
    from casquette.fleet.auth import read_token

    return read_token()


# How often to renew the HF OAuth access token from the stored refresh token.
# Below the token's ~1h lifetime and below fleet.auth._REFRESH_MARGIN_S, so a
# tick always lands inside the pre-expiry window. (Same as grabette.)
_TOKEN_REFRESH_INTERVAL_S = 600


async def _token_refresh_loop() -> None:
    from casquette.fleet.auth import get_hf_auth

    auth = get_hf_auth()
    while True:
        try:
            await asyncio.to_thread(auth.ensure_authenticated)
        except Exception:  # noqa: BLE001
            logger.debug("HF token refresh failed", exc_info=True)
        await asyncio.sleep(_TOKEN_REFRESH_INTERVAL_S)


def get_daemon_instance() -> Daemon | None:
    return _daemon


def _create_backend():
    """Create backend based on config (auto-detect, mock, or rpi)."""
    if settings.backend == "mock":
        from casquette.backend.mock import MockBackend
        logger.info("Using MockBackend (forced by config)")
        return MockBackend()
    elif settings.backend == "rpi":
        from casquette.backend.rpi import RpiBackend
        logger.info("Using RpiBackend (forced by config)")
        return RpiBackend(imu_hz=settings.imu_hz, imu_candidates=settings.imu_candidates)
    else:  # auto
        try:
            from casquette.backend.rpi import RpiBackend
            import picamera2  # noqa: F401
            logger.info("RPi hardware detected, using RpiBackend")
            return RpiBackend(imu_hz=settings.imu_hz, imu_candidates=settings.imu_candidates)
        except ImportError:
            from casquette.backend.mock import MockBackend
            logger.info("No RPi hardware, using MockBackend")
            return MockBackend()


async def _handle_relay_command(cmd: dict) -> dict:
    """Map fleet commands to casquette daemon actions (recording subset).

    Casquette is a pure fleet PEER (no physical button): it receives group
    start/stop from the fleet and records in lockstep via the shared
    capture_scheduler (same T0 mechanism grabette uses). upload_episodes pushes
    this device's streams to the shared raw dataset; process_dataset (driving the
    processing Space) is the one remaining deferred command.
    """
    from casquette.fleet.cancel import get_cancel_registry
    from casquette.fleet.capture_scheduler import get_capture_scheduler
    from casquette.task import episode_id_for, get_task_manager

    ctype = cmd.get("type")
    cmd_id = cmd.get("id")
    cancels = get_cancel_registry()

    if ctype == "cancel_dataset":
        # Above the daemon check on purpose: cancelling touches no capture
        # hardware and must work even when the daemon is down.
        args = cmd.get("args", {})
        marked = cancels.cancel(args.get("command_ids") or [])
        return {"status": "ok", "cancelled": marked, "job_id": args.get("job_id", "")}

    daemon = get_daemon_instance()
    if daemon is None:
        return {"status": "error", "message": "daemon not running"}
    scheduler = get_capture_scheduler()

    if ctype == "get_state":
        status = daemon.status
        if scheduler.is_scheduled():
            status["scheduled_start_utc"] = scheduler.scheduled_start_utc.isoformat()
        return {"status": "ok", "state": status}

    if ctype == "logout":
        # Drops the refresh token too: otherwise the refresh loop would log
        # the device straight back in.
        from casquette.fleet.auth import get_hf_auth
        get_hf_auth().delete_token()
        return {"status": "ok"}

    if ctype == "delete_episode":
        eid = (cmd.get("args") or {}).get("episode_id")
        if not eid:
            return {"status": "error", "message": "episode_id is required"}
        try:
            get_task_manager().delete_episode(eid)
            return {"status": "ok", "deleted": eid}
        except FileNotFoundError:
            return {"status": "ok", "deleted": None, "note": "not present on this device"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": str(e)}

    if ctype == "edit_task":
        args = cmd.get("args") or {}
        name = args.get("name")
        if not name:
            return {"status": "error", "message": "name is required"}
        get_task_manager().rename_task(
            name, new_name=args.get("new_name"), description=args.get("description"),
            device_signature=args.get("device_signature"),
        )
        return {"status": "ok"}

    if ctype == "delete_task":
        name = (cmd.get("args") or {}).get("name")
        if not name:
            return {"status": "error", "message": "name is required"}
        deleted = get_task_manager().delete_task_by_name(name)
        return {"status": "ok", "deleted": name if deleted else None}

    if ctype == "set_episode_members":
        # Fleet backfill of role→{device_id,name} on episodes. Batch or single.
        args = cmd.get("args") or {}
        tm = get_task_manager()
        sig = args.get("device_signature")
        if isinstance(args.get("episodes"), list):
            updated = tm.set_episodes_members(args["episodes"], device_signature=sig)
            return {"status": "ok", "updated": updated}
        eid = args.get("episode_id")
        if not eid:
            return {"status": "error", "message": "episode_id or episodes is required"}
        updated = tm.set_episode_members(eid, args.get("members") or {}, device_signature=sig)
        return {"status": "ok", "updated": 1 if updated else 0}

    if ctype == "upload_episodes":
        # Push THIS device's recorded streams for the given episodes into a
        # shared raw dataset, each under "{episode_id}/{role}" so peers' streams
        # for the same episode don't collide. Needs no capture hardware. Uploads
        # run in a thread so the relay keeps heartbeating; per-batch cancel
        # checkpoints let a fleet cancel_dataset stop it (see fleet/cancel.py).
        from casquette.fleet.hf import get_hf_client

        args = cmd.get("args", {})
        raw_repo = args.get("raw_repo")
        role = args.get("role")
        episode_ids = args.get("episode_ids") or []
        private = bool(args.get("private", False))
        if not raw_repo or not role:
            return {"status": "error", "message": "raw_repo and role are required"}
        tm = get_task_manager()
        hf = get_hf_client()
        uploaded, missing = [], []
        present = []
        for eid in episode_ids:
            ep_dir = tm.episode_dir(eid)
            if not ep_dir.exists():
                missing.append(eid)
                continue
            present.append((eid, ep_dir))

        # Skip episodes already on the Hub (saves hashing). Best-effort.
        if present and not cancels.is_cancelled(cmd_id):
            try:
                on_hub = await asyncio.to_thread(
                    hf.already_uploaded,
                    [(ep_dir, f"{eid}/{role}") for eid, ep_dir in present], raw_repo)
                uploaded.extend(eid for eid, _ in present if f"{eid}/{role}" in on_hub)
                present = [(eid, d) for eid, d in present if f"{eid}/{role}" not in on_hub]
            except Exception as e:  # noqa: BLE001
                logger.warning("could not list %s, uploading every episode: %s",
                               raw_repo, _exc_text(e))

        n_batches = -(-len(present) // _UPLOAD_BATCH_EPISODES)
        for start in range(0, len(present), _UPLOAD_BATCH_EPISODES):
            batch = present[start:start + _UPLOAD_BATCH_EPISODES]
            message = f"batch {start // _UPLOAD_BATCH_EPISODES + 1}/{n_batches} ({role})"
            if cancels.is_cancelled(cmd_id):
                return {"status": "cancelled", "role": role,
                        "uploaded": uploaded, "missing": missing}
            try:
                await asyncio.to_thread(
                    hf.upload_episodes,
                    [(ep_dir, f"{eid}/{role}") for eid, ep_dir in batch],
                    raw_repo, private, message)
                uploaded.extend(eid for eid, _ in batch)
            except Exception as e:  # noqa: BLE001
                if cancels.is_cancelled(cmd_id):
                    return {"status": "cancelled", "role": role,
                            "uploaded": uploaded, "missing": missing}
                which = batch[0][0] if len(batch) == 1 else f"{batch[0][0]}…{batch[-1][0]}"
                detail = _exc_text(e)
                if _is_rate_limited(e):
                    detail = ("Hugging Face rate limit reached — it resets on the "
                              "scale of an hour. Episodes already uploaded are kept. "
                              f"Hub said: {detail}")
                return {"status": "error",
                        "message": f"upload failed for {which}: {detail}",
                        "uploaded": uploaded, "missing": missing, "role": role}
        if cancels.is_cancelled(cmd_id):
            return {"status": "cancelled", "role": role,
                    "uploaded": uploaded, "missing": missing}
        return {"status": "ok", "role": role, "uploaded": uploaded, "missing": missing}

    if daemon.state != DaemonState.RUNNING:
        return {"status": "error", "message": f"daemon not ready ({daemon.state.value})"}

    backend = daemon.backend

    if ctype == "prepare_capture":
        # Warm the hardware ahead of a synchronized start. No-op/fast on casquette
        # (camera stays warm across captures).
        await backend.prepare_capture()
        return {"status": "ok"}

    if ctype == "start_capture":
        if backend.is_capturing:
            return {"status": "error", "message": "already capturing"}
        if scheduler.is_scheduled():
            return {"status": "error", "message": "a start is already scheduled"}
        # A known hardware fault fails the start NOW, in the command result the
        # fleet shows ("did not start: …"), instead of acking "scheduled" and
        # failing at T0 where only the journal sees it.
        if backend.hardware_error:
            return {"status": "error", "message": backend.hardware_error}
        tm = get_task_manager()
        args = cmd.get("args", {})
        task_name = args.get("task_name")
        task_id = tm.get_or_create_task(task_name) if task_name else args.get("task_id")
        start_at_utc = args.get("start_at_utc")
        members = args.get("members")
        signature = args.get("signature")

        # Resolve T0 BEFORE creating the episode: a group-synchronized start
        # derives the episode id from the shared T0 (episode_id_for), so every
        # device's episode folder for this recording has the same name.
        target = None
        if start_at_utc:
            try:
                target = datetime.fromisoformat(start_at_utc)
            except ValueError:
                return {"status": "error", "message": f"invalid start_at_utc: {start_at_utc!r}"}
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            late_s = (datetime.now(timezone.utc) - target).total_seconds()
            if late_s > MAX_START_LATENESS_S:
                return {"status": "error",
                        "message": f"start_at_utc is {late_s:.1f}s late (> {MAX_START_LATENESS_S}s); refusing"}
            if late_s > 0:
                logger.warning("scheduled start %.2fs late; starting best-effort", late_s)

        episode_id = tm.create_episode(
            task_id,
            episode_id=episode_id_for(target) if target else None,
            members=members,
            signature=signature,
        )
        episode_dir = tm.episode_dir(episode_id)

        if target is None:
            try:
                await backend.start_capture(episode_dir)
            except Exception:
                tm.discard_pending_episode()
                raise
            return {"status": "ok", "episode_id": episode_id}

        # Scheduled (synchronized group) start: wait for T0 in the background
        # and ack immediately so the fleet dispatch round-trip doesn't block.
        await scheduler.schedule(backend, tm, episode_dir, target)
        return {"status": "scheduled", "episode_id": episode_id, "start_at_utc": target.isoformat()}

    if ctype == "stop_capture":
        tm = get_task_manager()
        try:
            outcome = await scheduler.cancel_or_wait(backend)
        except RuntimeError as e:
            return {"status": "error", "message": str(e)}
        if outcome == "cancelled":
            tm.discard_pending_episode()
            return {"status": "cancelled"}
        if not backend.is_capturing:
            return {"status": "error", "message": "not capturing"}

        args = cmd.get("args", {})
        stop_at_utc = args.get("stop_at_utc")
        if stop_at_utc:
            try:
                stop_target = datetime.fromisoformat(stop_at_utc)
            except ValueError:
                return {"status": "error", "message": f"invalid stop_at_utc: {stop_at_utc!r}"}
            if stop_target.tzinfo is None:
                stop_target = stop_target.replace(tzinfo=timezone.utc)
            await scheduler.schedule_stop(backend, tm, stop_target)
            return {"status": "scheduled_stop", "stop_at_utc": stop_target.isoformat()}

        result = await backend.stop_capture()
        tm.register_episode(getattr(result, "episode_id", None))
        # Return a plain dict (the relay json.dumps this — a pydantic object
        # would raise TypeError and kill the relay loop).
        return {"status": "ok",
                "result": result.model_dump() if hasattr(result, "model_dump") else result}

    return {"status": "error", "message": f"unknown command '{ctype}'"}


async def _loop_lag_watchdog(threshold_s: float = 0.5, tick_s: float = 0.1) -> None:
    """Diagnostic: log whenever the event loop / main thread stalls.

    Sleeps tick_s and measures the ACTUAL elapsed time; the excess over tick_s
    is wall-time the main thread could not run — either a non-yielding coroutine
    blocked the loop, or the GIL was held by another thread. A stall >threshold_s
    is long enough to stop picamera2 recycling buffers → the CSI "Dequeue timer
    expired" / frontend-timeout wedge. Logged at WARNING with the lag and a
    wall-clock so it can be lined up against a camera timeout in the journal.

    Opt-in via CASQUETTE_LOOP_WATCHDOG=1 — pure diagnostic, negligible cost.
    """
    loop = asyncio.get_running_loop()
    if os.getenv("CASQUETTE_ASYNCIO_DEBUG"):
        # asyncio debug mode logs "Executing <callback> took N seconds" for any
        # callback exceeding slow_callback_duration — this NAMES what blocks the
        # loop, turning an anonymous stall into a specific culprit. 0.5s keeps
        # the log to only the significant blockers.
        loop.set_debug(True)
        loop.slow_callback_duration = 0.5
        logger.info("asyncio debug on (slow_callback_duration=0.5s)")
    while True:
        t0 = loop.time()
        await asyncio.sleep(tick_s)
        lag = loop.time() - t0 - tick_s
        if lag > threshold_s:
            logger.warning(
                "EVENT-LOOP STALL: main thread blocked %.2fs (at %s)",
                lag, _time.strftime("%H:%M:%S"),
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _daemon

    backend = _create_backend()
    _daemon = Daemon(backend)
    await _daemon.start()

    if os.getenv("CASQUETTE_LOOP_WATCHDOG"):
        asyncio.create_task(_loop_lag_watchdog())
        logger.info("Loop-lag watchdog enabled (CASQUETTE_LOOP_WATCHDOG=1)")

    # HF auth: an OAuth login's access token lasts ~1h, so renew it from the
    # stored refresh token now (covers a reboot) and every 10 min (covers long
    # uptimes). No-op for a manual token. fleet.auth stays clear of
    # huggingface_hub, keeping that heavy import out of the steady state.
    token_refresh_task = None
    if settings.relay_enabled:
        token_refresh_task = asyncio.create_task(_token_refresh_loop())
        if not _read_hf_token():
            logger.warning(
                "no HF token — fleet relay will idle until one is set "
                "(sign in from the dashboard, or run `huggingface-cli login` on the device)"
            )

    # Fleet relay loop. Casquette is a pure peer — NO button listener, and no
    # PiSugar battery / hand providers (grabette-only hardware).
    relay_task = None
    if settings.relay_enabled:
        from casquette.fleet.relay_client import RelayClient
        from casquette.task import get_task_manager

        def _device_activity() -> str:
            """idle | capturing — casquette has no upload/process jobs yet."""
            try:
                d = get_daemon_instance()
                if d is not None and d.state == DaemonState.RUNNING and d.backend.is_capturing:
                    return "capturing"
            except Exception:
                pass
            return "idle"

        def _last_episode_report():
            """(episode id, issues) of the last take, once settled."""
            d = get_daemon_instance()
            if d is None or d.backend.is_capturing:
                return None
            return tm.last_episode_issues()

        tm = get_task_manager()
        relay = RelayClient(
            base_url=settings.relay_url,
            token_provider=_read_hf_token,
            device_id=settings.device_id,
            name=settings.device_name,
            capabilities=["get_state", "start_capture", "stop_capture", "prepare_capture",
                          "cancel_dataset", "delete_episode", "edit_task", "delete_task",
                          "upload_episodes", "logout"],
            tasks_provider=tm.report_tasks,
            tasks_rev_provider=tm.revision,
            activity_provider=_device_activity,
            # Hardware faults (no IMU): the device refuses to record, and the
            # fleet must know BEFORE a start, not discover it from missing data.
            fault_provider=lambda: getattr(_daemon.backend, "hardware_error", "") if _daemon else "",
            last_episode_provider=_last_episode_report,
        )
        relay_task = asyncio.create_task(relay.run(_handle_relay_command))
        logger.info("Relay started → %s (device: %s)", settings.relay_url, settings.device_id)

    yield

    for task in (relay_task, token_refresh_task):
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    await _daemon.stop()
    _daemon = None


def create_app() -> FastAPI:
    from casquette.app.routers.camera import router as camera_router
    from casquette.app.routers.daemon import router as daemon_router
    from casquette.app.routers.hf_auth import router as hf_auth_router
    from casquette.app.routers.state import router as state_router
    from casquette.app.routers.system import router as system_router
    from casquette.app.routers.tasks import router as tasks_router
    from casquette.app.routers.ui import router as ui_router
    from casquette.app.routers.wifi import router as wifi_router

    app = FastAPI(
        title="Casquette",
        description="POV camera data collection service",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    app.include_router(daemon_router)
    app.include_router(state_router)
    app.include_router(camera_router)
    app.include_router(system_router)
    app.include_router(tasks_router)
    app.include_router(wifi_router)
    app.include_router(hf_auth_router)
    # The dashboard (Overview / Episodes / Network) at "/".
    app.include_router(ui_router)

    return app
