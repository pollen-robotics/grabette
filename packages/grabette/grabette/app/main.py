from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from grabette.config import settings
from grabette.daemon import Daemon

# Route Gradio's internal file cache to the SD card. Gradio copies every
# callback-returned file path into its cache dir (defaulting to
# tempfile.gettempdir() + '/gradio/' = /tmp/gradio/ on Pi OS) to serve it
# via a stable /gradio_api/file=... URL. Without this, downloading a
# multi-GB episode archive fills the /tmp tmpfs even though we already
# routed archive staging + api_client staging to the SD card. Must be set
# before any Gradio import so the cache-dir constant is picked up.
os.environ.setdefault("GRADIO_TEMP_DIR", str(settings.data_dir / ".gradio-cache"))
(settings.data_dir / ".gradio-cache").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Maximum tolerated start lateness for a scheduled (group) start. If T0 has
# already passed by less than this when the command is processed, we start
# immediately (best-effort) — a tiny delivery delay shouldn't drop the episode.
# Beyond it we REFUSE: a start later than this is too desynced to be usable for
# multi-device data, and keeping it would produce an episode that's still paired
# by episode_id but misaligned (a false pair) — worse than an honest miss.
MAX_START_LATENESS_S = 1.0

# How often to refresh the HF OAuth access token from the stored refresh token.
# Must stay below the token's ~1h lifetime and below auth._REFRESH_MARGIN_S so a
# tick always lands inside the pre-expiry refresh window.
_TOKEN_REFRESH_INTERVAL_S = 600  # 10 min

_daemon: Daemon | None = None


def get_daemon_instance() -> Daemon | None:
    return _daemon


def _create_backend():
    """Create backend based on config (auto-detect, mock, or rpi)."""
    if settings.backend == "mock":
        from grabette.backend.mock import MockBackend
        logger.info("Using MockBackend (forced by config)")
        return MockBackend()
    elif settings.backend == "rpi":
        from grabette.backend.rpi import RpiBackend
        logger.info("Using RpiBackend (forced by config)")
        return RpiBackend(
            enable_angle=settings.angle_sensors,
            enable_oakd=settings.enable_oakd,
            oakd_keepalive_s=settings.oakd_keepalive_s,
            depth_camera=settings.depth_camera,
            orbbec_ir_exposure_us=settings.orbbec_ir_exposure_us,
            orbbec_ir_gain=settings.orbbec_ir_gain,
        )
    else:  # auto
        try:
            from grabette.backend.rpi import RpiBackend
            import picamera2  # noqa: F401
            logger.info("RPi hardware detected, using RpiBackend")
            return RpiBackend(
                enable_angle=settings.angle_sensors,
                enable_oakd=settings.enable_oakd,
                oakd_keepalive_s=settings.oakd_keepalive_s,
                depth_camera=settings.depth_camera,
                orbbec_ir_exposure_us=settings.orbbec_ir_exposure_us,
                orbbec_ir_gain=settings.orbbec_ir_gain,
            )
        except ImportError:
            from grabette.backend.mock import MockBackend
            logger.info("No RPi hardware, using MockBackend")
            return MockBackend()


_button_listener = None

# --- processing Space (raw → LeRobot conversion) ------------------------------
# Talking to the Space has three very different latency regimes, so ONE timeout
# for all of them is wrong: a single 60s cap used to apply to the whole session,
# which meant a cold start (minutes) was reported as a failed call. A sleeping or
# rebuilding HF Space does not refuse the connection — the HF proxy HOLDS it while
# the container boots — so the wake gets its own budget, outside the per-request
# timeouts. The URL is never hardcoded here: the fleet names the Space in the
# command's args, and the wake follows whatever it names.
_SPACE_WAKE_BUDGET_S = 300.0        # how long a cold start may take before giving up
_SPACE_WAKE_PROBE_TIMEOUT_S = 20.0  # one liveness probe
_SPACE_WAKE_RETRY_S = 5.0           # between probes
_SPACE_POST_TIMEOUT_S = 120.0       # /api/process, on a Space known to be awake
_SPACE_POLL_TIMEOUT_S = 30.0        # one /api/status poll (retried on failure)

# Probing the app is the authoritative readiness test, but it cannot tell apart
# "still booting" from two states it will never get out of. When the fleet also
# names the Space REPO (space_repo, optional), the Hub runtime resolves both:
#   - a broken Space would otherwise burn the whole wake budget for nothing;
#   - a PAUSED/STOPPED Space does NOT wake on incoming traffic at all, so no
#     amount of probing helps — it needs an explicit restart.
# The repo id can't be derived from the .hf.space URL (org names may contain
# hyphens), which is why it's a separate arg rather than something we compute.
_SPACE_FAILED_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}
_SPACE_HALTED_STAGES = {"PAUSED", "STOPPED"}
# The stage read is a second network round-trip, so it must not run on every
# probe: once up front (to fail fast), then occasionally to catch a Space that
# breaks while we wait.
_SPACE_STAGE_RECHECK_PROBES = 6


class _CommandCancelled(Exception):
    """The operator cancelled this command while it waited on something."""


def _exc_text(e: BaseException) -> str:
    """Never-empty description of an exception.

    `str(asyncio.TimeoutError())` is the EMPTY STRING, so interpolating an
    exception straight into a message can report only "processing failed:" —
    leaving the operator nothing to act on, and the failure that says the least is
    the most common one. Fall back to the class name, which at least names the
    failure mode.
    """
    return str(e).strip() or type(e).__name__


def _space_stage(space_repo: str, token: str | None) -> str | None:
    """Current SpaceStage (e.g. 'RUNNING', 'SLEEPING'), or None if unreadable.

    Best effort only: a token without access to the Space repo, or an unreachable
    Hub, yields None. The caller then falls back to probing the app itself, which
    is the authoritative readiness test anyway — so this can only ever make the
    wake smarter, never break it.
    """
    try:
        from huggingface_hub import get_space_runtime

        return get_space_runtime(space_repo, token=token or None).stage
    except Exception as e:  # noqa: BLE001 — enrichment, never fatal
        logger.info("Could not read Space runtime for %s: %s", space_repo, _exc_text(e))
        return None


def _restart_space(space_repo: str, token: str | None) -> bool:
    """Ask the Hub to restart a halted Space. True if the call went through.

    Needs WRITE access to the Space repo. A read-only token just fails here and
    we fall back to reporting the manual step to the operator.
    """
    try:
        from huggingface_hub import restart_space

        restart_space(space_repo, token=token or None)
        return True
    except Exception as e:  # noqa: BLE001
        logger.info("Could not restart Space %s: %s", space_repo, _exc_text(e))
        return False


async def _wake_space(session, space_url: str, headers: dict,
                      is_cancelled=None, space_repo: str | None = None,
                      token: str | None = None,
                      probe_timeout=None) -> str | None:
    """Wait until the processing Space really serves requests. None once awake,
    else the reason to report.

    The probe is GET /api/status/<throwaway id>: cheap, needs no job, and it is a
    route of the app ITSELF — so a JSON answer proves the app is up, not just the
    proxy in front of it. That distinction is the whole point: a container still
    booting answers with an HTML holding page, which must not be mistaken for
    "awake". Headers are passed through so this works on a private Space too.

    When space_repo is given, the Hub runtime is consulted as well: it turns a
    doomed wait into an immediate, named failure, and restarts a Space that is
    halted rather than merely asleep. Both are best effort — without the repo, or
    with a token that can't read it, the wake behaves exactly as before.

    Raises _CommandCancelled if the build is cancelled while we wait: this wait
    can last minutes, and a conversion nobody wants any more must not start.

    session and probe_timeout are both injected, so this function never touches
    the HTTP library itself — the caller owns that choice, and the wake logic
    stays unit-testable against a stub session with no aiohttp installed.
    """
    deadline = time.monotonic() + _SPACE_WAKE_BUDGET_S
    last, probes = "no response", 0
    restart_tried = False
    while time.monotonic() < deadline:
        if is_cancelled and is_cancelled():
            raise _CommandCancelled
        probes += 1

        if space_repo and (probes == 1 or probes % _SPACE_STAGE_RECHECK_PROBES == 0):
            stage = await asyncio.to_thread(_space_stage, space_repo, token)
            if stage in _SPACE_FAILED_STAGES:
                return (f"the processing Space at {space_url} is in a failed state "
                        f"(stage={stage}) — it will not start on its own. Check its "
                        f"build/runtime logs on Hugging Face.")
            if stage in _SPACE_HALTED_STAGES and not restart_tried:
                # A halted Space ignores incoming traffic, so probing alone would
                # burn the whole budget. Try once; if the token is read-only the
                # wait continues and the timeout message names the manual step.
                restart_tried = True
                if await asyncio.to_thread(_restart_space, space_repo, token):
                    logger.info(
                        "processing Space was %s — restart requested", stage,
                    )
                else:
                    return (f"the processing Space at {space_url} is {stage} and this "
                            f"device's token cannot restart it. Restart it once on "
                            f"Hugging Face, then retry.")

        try:
            async with session.get(
                f"{space_url}/api/status/_wake", headers=headers,
                timeout=probe_timeout,
            ) as r:
                ctype = r.headers.get("Content-Type", "")
                await r.read()  # drain so the connection can be reused
                if r.status == 200 and "json" in ctype.lower():
                    if probes > 1:
                        logger.info("processing Space awake after %d probe(s)", probes)
                    return None
                last = (f"HTTP {r.status}" if r.status != 200
                        else f"still starting (content-type {ctype or 'unknown'})")
        except Exception as e:  # noqa: BLE001 — almost certainly still booting; retry
            last = _exc_text(e)
        logger.info("processing Space not ready yet (%s) — retrying", last)
        await asyncio.sleep(_SPACE_WAKE_RETRY_S)
    return (f"the processing Space at {space_url} did not wake up within "
            f"{int(_SPACE_WAKE_BUDGET_S)}s after {probes} probe(s) — last: {last}. "
            f"Open it once in a browser, then retry.")


async def _handle_relay_command(cmd: dict) -> dict:
    """Map fleet commands to grabette daemon actions.

    start_capture/stop_capture share their scheduling state machine (see
    capture_scheduler.py) with the physical button and the local UI, so a
    fleet-dispatched synchronized start (start_at_utc set by a group's
    /api/fleet/groups/{id}/start_capture or another device's local trigger)
    behaves identically to one triggered locally on this device.
    """
    from grabette.cancel import get_cancel_registry
    from grabette.capture_scheduler import get_capture_scheduler
    from grabette.daemon import DaemonState
    from grabette.task import episode_id_for
    from grabette.app.routers.tasks import get_task_manager

    ctype = cmd.get("type")
    cmd_id = cmd.get("id")
    cancels = get_cancel_registry()

    if ctype == "cancel_dataset":
        # Handled ABOVE the daemon check on purpose: cancelling touches no capture
        # hardware, and it must still work when the daemon is down — that is
        # precisely a moment when a stuck upload needs stopping.
        #
        # The operator cancelled a fleet dataset build: flag the command ids it
        # names so the running (or still-queued) upload/processing handler stops at
        # its next checkpoint. See grabette.cancel for why this goes through a
        # registry instead of cancelling a task directly, and why the relay must
        # dispatch this command on its FAST PATH (relay_client._FAST_PATH_TYPES) —
        # it must never wait for the work it is cancelling.
        #
        # It deliberately does NOT touch args["raw_repo"]: the partially-uploaded
        # raw dataset is left in place (the fleet passes keep_raw for exactly this),
        # so a cancelled build can be inspected or re-run without re-uploading.
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
        from huggingface_hub import logout as hf_logout
        hf_logout()
        return {"status": "ok"}

    if ctype == "upload_episodes":
        # Fleet-orchestrated dataset build: push THIS device's recorded streams
        # for the given episodes into a shared raw dataset, each under
        # "{episode_id}/{role}" so peers' streams for the same episode don't
        # collide. Needs no capture hardware, so it runs regardless of daemon
        # state. Uploads run in a thread so the relay keeps heartbeating.
        from grabette.app.routers.huggingface import get_hf_client

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
        for eid in episode_ids:
            # Cancellation checkpoint, once per episode. Checked BEFORE the first
            # upload too: the cancel may have landed while this command was still
            # queued behind another one (the relay's worker is serial), in which
            # case nothing should be uploaded at all. Per-episode is the finest
            # granularity available — one episode is a single blocking
            # api.upload_folder inside hf.upload_episode, which cannot be
            # interrupted from outside.
            if cancels.is_cancelled(cmd_id):
                return {"status": "cancelled", "role": role,
                        "uploaded": uploaded, "missing": missing}
            ep_dir = tm.episode_dir(eid)
            if not ep_dir.exists():
                missing.append(eid)
                continue
            try:
                await asyncio.to_thread(hf.upload_episode, ep_dir, raw_repo, None, f"{eid}/{role}", private)
                uploaded.append(eid)
            except Exception as e:  # noqa: BLE001
                if cancels.is_cancelled(cmd_id):
                    # Cancelled mid-upload: the failure is a consequence, not news.
                    return {"status": "cancelled", "role": role,
                            "uploaded": uploaded, "missing": missing}
                return {"status": "error", "message": f"upload failed for {eid}: {_exc_text(e)}",
                        "uploaded": uploaded, "missing": missing, "role": role}
        if cancels.is_cancelled(cmd_id):
            return {"status": "cancelled", "role": role,
                    "uploaded": uploaded, "missing": missing}
        return {"status": "ok", "role": role, "uploaded": uploaded, "missing": missing}

    if ctype == "process_dataset":
        # Fleet-orchestrated dataset build, processing step. THIS device triggers
        # the processing Space with ITS OWN long-lived HF token and polls it to
        # completion, then reports the result. Done device-side (not by the
        # fleet) so no HF token is ever cached on or forwarded through the fleet
        # — same device→Space trust boundary the SLAM flow already uses. Needs no
        # capture hardware. The command completes only when the Space is done.
        import aiohttp
        from huggingface_hub import get_token

        args = cmd.get("args", {})
        space_url = (args.get("space_url") or "").rstrip("/")
        source_repo = args.get("source_repo")
        target_repo = args.get("target_repo")
        if not space_url or not source_repo or not target_repo:
            return {"status": "error", "message": "space_url, source_repo, target_repo are required"}
        token = get_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        payload = {
            "source_repo": source_repo, "target_repo": target_repo,
            "task": args.get("task") or target_repo.split("/")[-1],
            "roles": args.get("roles") or [], "private": bool(args.get("private", False)),
            # Leave the raw dataset alone after the conversion (fleet-controlled,
            # default True there). Passed through untouched so the Space's own
            # default never silently deletes a raw we were asked to keep.
            "keep_raw": bool(args.get("keep_raw", True)),
        }
        # Cancelled while this command sat in the relay's queue → never even start
        # the conversion (it would push a dataset nobody asked for any more).
        if cancels.is_cancelled(cmd_id):
            return {"status": "cancelled"}
        try:
            # No session-wide timeout: each request below sets its own, because
            # they have nothing in common (see the _SPACE_* constants).
            async with aiohttp.ClientSession() as s:
                # The Space may be asleep or rebuilding — wake it FIRST, on its own
                # budget, so a cold start is never reported as a failed conversion
                # (it used to surface as a bare, message-less timeout).
                # space_repo is optional: when the fleet sends it, the wake can
                # also fail fast on a broken Space and restart a halted one.
                not_awake = await _wake_space(
                    s, space_url, headers,
                    lambda: cancels.is_cancelled(cmd_id),
                    space_repo=args.get("space_repo"),
                    token=token,
                    probe_timeout=aiohttp.ClientTimeout(
                        total=_SPACE_WAKE_PROBE_TIMEOUT_S
                    ),
                )
                if not_awake:
                    return {"status": "error", "message": not_awake}
                async with s.post(f"{space_url}/api/process", json=payload, headers=headers,
                                  timeout=aiohttp.ClientTimeout(total=_SPACE_POST_TIMEOUT_S)) as r:
                    if r.status != 200:
                        return {"status": "error",
                                "message": f"Space /api/process HTTP {r.status}: {(await r.text())[:200]}"}
                    space_jid = (await r.json()).get("job_id")
                if not space_jid:
                    return {"status": "error", "message": "Space returned no job_id"}
                deadline = time.monotonic() + 3600.0  # 60-min cap
                while True:
                    if time.monotonic() > deadline:
                        return {"status": "error", "message": "processing timed out"}
                    await asyncio.sleep(5.0)
                    # Stop waiting on the Space as soon as the build is cancelled.
                    # NB: this releases the device, it does not stop the Space —
                    # /api/process has no cancel endpoint, so a conversion already
                    # under way runs to completion and may still push its target
                    # dataset. The raw is kept either way (keep_raw).
                    if cancels.is_cancelled(cmd_id):
                        return {"status": "cancelled", "space_job_id": space_jid}
                    try:
                        async with s.get(f"{space_url}/api/status/{space_jid}", headers=headers,
                                         timeout=aiohttp.ClientTimeout(total=_SPACE_POLL_TIMEOUT_S)) as r:
                            if r.status != 200:
                                continue
                            st = await r.json()
                    except Exception:  # noqa: BLE001 — transient poll error, retry
                        continue
                    sstatus = st.get("status", "running")
                    if sstatus == "done":
                        return {"status": "ok", "result_url": st.get("result")}
                    if sstatus in ("error", "not_found"):
                        # not_found = the Space forgot the job. Its job list lives in
                        # memory, so this means it restarted mid-conversion (OOM, a
                        # redeploy) — say so, rather than echoing "not_found".
                        lost = (f"the Space no longer knows job {space_jid} — it restarted "
                                f"mid-conversion (its job list is kept in memory)")
                        return {"status": "error",
                                "message": st.get("error") or (
                                    lost if sstatus == "not_found" else "the Space reported a failure")}
        except _CommandCancelled:
            return {"status": "cancelled"}
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": f"processing failed: {_exc_text(e)}"}

    if ctype == "delete_episode":
        # Fleet "delete last episode" / "discard these takes": remove this device's
        # local files + registry entries. Needs no capture hardware. Absent locally
        # (never recorded here / already gone) → success (idempotent).
        #
        # Accepts a batch ("episode_ids": [...]) as well as a single episode_id,
        # like set_episode_members does: discarding a triage selection is one
        # operator gesture, and one command per episode would queue fifty of them
        # through a serial relay worker.
        from grabette.app.routers.tasks import get_task_manager

        args = cmd.get("args") or {}
        eids = args.get("episode_ids")
        if not isinstance(eids, list):
            eids = [args.get("episode_id")] if args.get("episode_id") else []
        if not eids:
            return {"status": "error", "message": "episode_id or episode_ids is required"}
        tm = get_task_manager()
        deleted, absent, errors = [], [], []
        for eid in eids:
            try:
                tm.delete_episode(eid)
                deleted.append(eid)
            except FileNotFoundError:
                absent.append(eid)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{eid}: {e}")
        if errors:
            return {"status": "error", "message": "; ".join(errors),
                    "deleted": deleted, "absent": absent}
        return {"status": "ok", "deleted": deleted, "absent": absent}

    if ctype == "edit_task":
        # Fleet task edit (rename / re-describe), keyed by task name. Idempotent.
        from grabette.app.routers.tasks import get_task_manager

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
        # Fleet task delete: remove the task AND its recorded episodes locally,
        # by name. Idempotent (absent → success). Needs no capture hardware.
        from grabette.app.routers.tasks import get_task_manager

        name = (cmd.get("args") or {}).get("name")
        if not name:
            return {"status": "error", "message": "name is required"}
        deleted = get_task_manager().delete_task_by_name(name)
        return {"status": "ok", "deleted": name if deleted else None}

    if ctype == "assign_episodes":
        # Fleet "file these recordings under task X", keyed by task NAME (local ids
        # mean nothing across devices). Serves both triaging the unassigned inbox
        # and repairing a split — a fleet-wide dispatch that makes every device
        # agree on one task name for the episode. Idempotent, and safe to send to a
        # device that never had these episodes: move_episodes skips ids it knows
        # nothing about rather than inventing them.
        from grabette.app.routers.tasks import get_task_manager

        args = cmd.get("args") or {}
        name = (args.get("task_name") or "").strip()
        episode_ids = args.get("episode_ids") or []
        if not name:
            return {"status": "error", "message": "task_name is required"}
        if not episode_ids:
            return {"status": "error", "message": "episode_ids is required"}
        tm = get_task_manager()
        result = tm.move_episodes(episode_ids, tm.get_or_create_task(name))
        return {"status": "ok", "task": name, **result}

    if ctype == "set_episode_members":
        # Fleet "fill devices": backfill who recorded episodes (role →
        # {device_id, name}) saved before members were persisted. Accepts a batch
        # ("episodes": [{episode_id, members}, …]) or a single episode_id/members.
        # Merges (partial repair) and is idempotent; absent locally → success.
        from grabette.app.routers.tasks import get_task_manager

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

    if daemon.state != DaemonState.RUNNING:
        return {"status": "error", "message": f"daemon not ready ({daemon.state.value})"}

    backend = daemon.backend
    if ctype == "prepare_capture":
        # Warm the hardware ahead of a synchronized start (fleet dispatches
        # this when a session opens). No-op/fast when already warm.
        await backend.prepare_capture()
        return {"status": "ok"}
    if ctype == "start_capture":
        if backend.is_capturing:
            return {"status": "error", "message": "already capturing"}
        if scheduler.is_scheduled():
            return {"status": "error", "message": "a start is already scheduled"}
        tm = get_task_manager()
        args = cmd.get("args", {})
        task_name = args.get("task_name")
        task_id = tm.get_or_create_task(task_name) if task_name else args.get("task_id")
        start_at_utc = args.get("start_at_utc")
        # Who's recording this episode (role → {device_id, name}) and the task's
        # device signature, sent by the fleet. Persisted with the episode so this
        # device can later name its peers even when they're offline.
        members = args.get("members")
        signature = args.get("signature")

        # Resolve T0 BEFORE creating the episode: a group-synchronized start
        # derives the episode id from the shared T0 (see episode_id_for), not
        # from local wall-clock creation time, so every device's episode
        # folder for this recording has the same name — even though each one
        # actually creates its directory whenever it happens to process this
        # command (which can differ by up to the fleet poll interval).
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
                # Too late to be usable for multi-device data — refuse rather
                # than keep a desynced episode. (Common when the relay was busy
                # muxing the previous stop and delivered this command late.)
                return {"status": "error", "message": f"start_at_utc is {late_s:.1f}s late (> {MAX_START_LATENESS_S}s); refusing"}
            if late_s > 0:
                # Within tolerance: start immediately (the scheduler fires now
                # when T0 has just passed); sync metadata records the real start.
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

        # Synchronized (group) stop: wait out a shared T_stop in the background
        # and ack immediately, so every member ends its recording at the same
        # instant instead of the pressed device stopping now and peers lagging
        # by the fleet round-trip. Mirror of the scheduled start.
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
        # Return a plain dict, not the CaptureStatus model: the relay POSTs this
        # result as JSON, and json.dumps can't serialize a pydantic object — the
        # TypeError would escape the relay loop and kill it (the device then
        # vanishes from the fleet until its service is restarted).
        return {"status": "ok", "result": result.model_dump() if hasattr(result, "model_dump") else result}
    return {"status": "error", "message": f"unknown command '{ctype}'"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _daemon, _button_listener
    import asyncio

    backend = _create_backend()
    _daemon = Daemon(backend)
    await _daemon.start()

    # Start physical button listener on RPi
    if settings.button_enabled:
        try:
            from grabette.button_listener import ButtonListener
            from grabette.app.routers.tasks import get_task_manager

            _button_listener = ButtonListener(backend, get_task_manager())
            _button_listener.start(asyncio.get_running_loop())
        except Exception as e:
            logger.debug("Button listener not started: %s", e)
            _button_listener = None

    # Keep the device logged in as the last account without the operator redoing
    # OAuth: the HF OAuth access token is short-lived, so refresh it from the
    # stored refresh token at startup (covers a reboot) and periodically before
    # it expires (covers long uptimes). No-op if login was a manual PAT.
    refresh_task = None
    if settings.relay_enabled:
        from grabette.auth import get_hf_auth

        _hf_auth = get_hf_auth()
        try:
            await _hf_auth.ensure_authenticated()
        except Exception:
            logger.debug("startup HF token refresh failed", exc_info=True)

        async def _token_refresh_loop():
            while True:
                await asyncio.sleep(_TOKEN_REFRESH_INTERVAL_S)
                try:
                    await _hf_auth.ensure_authenticated()
                except Exception:
                    logger.debug("periodic HF token refresh failed", exc_info=True)

        refresh_task = asyncio.create_task(_token_refresh_loop())

    # Start fleet relay loop
    relay_task = None
    if settings.relay_enabled:
        from huggingface_hub import get_token
        from grabette.relay_client import RelayClient
        from grabette.app.routers.system import _pisugar_battery
        from grabette.app.routers.tasks import get_task_manager
        from grabette.daemon import DaemonState
        from grabette.jobs import JobStatus, get_job_manager

        def _device_activity() -> str:
            """Current device activity for the fleet heartbeat:
            idle | capturing | uploading | processing. Surfaces local, dashboard-
            initiated work (SLAM push / episode upload) that the fleet can't infer
            on its own, plus live capture. Best-effort & non-throwing — a bad read
            just reports 'idle'. All in-memory, so it's cheap on the heartbeat."""
            try:
                active = [j for j in get_job_manager().list_jobs()
                          if j.status in (JobStatus.PENDING, JobStatus.RUNNING)]
                if any(j.name.startswith("push:") for j in active):
                    return "processing"  # SLAM convert + push to the dataset
                if any(j.name.startswith("upload:") for j in active):
                    return "uploading"
            except Exception:
                pass
            try:
                d = get_daemon_instance()
                if d is not None and d.state == DaemonState.RUNNING and d.backend.is_capturing:
                    return "capturing"
            except Exception:
                pass
            return "idle"

        relay = RelayClient(
            base_url=settings.relay_url,
            token_provider=get_token,
            device_id=settings.device_id,
            name=settings.device_name,
            capabilities=["get_state", "start_capture", "stop_capture", "logout",
                          "upload_episodes", "process_dataset", "cancel_dataset",
                          "delete_episode", "edit_task", "delete_task",
                          "assign_episodes", "prepare_capture"],
            hand=settings.hand,
            battery_provider=_pisugar_battery,  # reported via heartbeat for the fleet UI
            tasks_provider=get_task_manager().report_tasks,  # this device's tasks, sent on connect
            # Loose episodes (recorded outside any task) — their own channel, so
            # the fleet can offer them for triage without them ever passing for tasks.
            unassigned_provider=get_task_manager().report_unassigned,
            tasks_rev_provider=get_task_manager().revision,  # re-report when tasks change
            activity_provider=_device_activity,  # device state, reported via heartbeat
        )
        relay_task = asyncio.create_task(relay.run(_handle_relay_command))
        logger.info("Relay started → %s (device: %s)", settings.relay_url, settings.device_id)

    yield

    import contextlib
    if relay_task is not None:
        relay_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await relay_task

    if refresh_task is not None:
        refresh_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await refresh_task

    if _button_listener is not None:
        _button_listener.stop()
        _button_listener = None
    await _daemon.stop()
    _daemon = None


def create_app() -> FastAPI:
    from grabette.app.routers.camera import router as camera_router
    from grabette.app.routers.daemon import router as daemon_router
    from grabette.app.routers.huggingface import router as hf_router
    from grabette.app.routers.tasks import router as tasks_router
    from grabette.app.routers.state import router as state_router
    from grabette.app.routers.system import router as system_router

    app = FastAPI(
        title="Grabette",
        description="Robotic manipulation data collection service",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins for dev / web app connectivity
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global error handler
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    from grabette.app.routers.charts import router as charts_router
    from grabette.app.routers.oakd import router as oakd_router
    from grabette.app.routers.replay import router as replay_router
    from grabette.app.routers.viewer import router as viewer_router
    from grabette.app.routers.wifi import router as wifi_router
    from grabette.app.routers.teleop import router as teleop_router

    app.include_router(daemon_router)
    app.include_router(state_router)
    app.include_router(wifi_router)
    app.include_router(tasks_router)
    app.include_router(camera_router)
    app.include_router(hf_router)
    app.include_router(system_router)
    app.include_router(viewer_router)
    app.include_router(charts_router)
    app.include_router(replay_router)
    app.include_router(teleop_router)
    app.include_router(oakd_router)

    # Serve URDF model + STL meshes as static files
    _urdf_dir = Path(__file__).resolve().parent.parent.parent / "urdf"
    if _urdf_dir.is_dir():
        app.mount("/urdf", StaticFiles(directory=str(_urdf_dir)), name="urdf")
        logger.info("URDF assets mounted at /urdf from %s", _urdf_dir)

    # Auth router (OAuth PKCE + manual token) — must be registered before Gradio
    from grabette.auth import get_hf_auth
    from grabette.webauth import build_auth_router

    app.include_router(build_auth_router(get_hf_auth()))

    # Mount Gradio UI if enabled and installed
    if settings.ui_enabled:
        try:
            import gradio as gr
            from grabette.ui.app import create_ui

            demo = create_ui()
            # `allowed_paths` whitelists directories from which Gradio's
            # `/gradio_api/file=<path>` handler will serve files to the
            # browser. Without this, the multi-GB episode archives written
            # to data_dir/.downloads/ pass silently through — the file
            # exists on disk but Gradio refuses to hand it out, and the UI
            # never surfaces a download link. Default allowed paths cover
            # only Gradio's own cache and the OS temp dir.
            app = gr.mount_gradio_app(
                app, demo, path="/",
                allowed_paths=[str(settings.data_dir / ".downloads")],
            )
            logger.info("Gradio UI mounted at /")
        except ImportError:
            logger.warning(
                "Gradio not installed, UI disabled "
                "(install with: uv sync --extra ui)"
            )

    return app
