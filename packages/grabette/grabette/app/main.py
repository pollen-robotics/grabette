from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
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
from grabette.errors import exc_text as _exc_text

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

# --- episode upload (raw dataset) ---------------------------------------------
# Three layers, because each catches what the one below it cannot:
#
# 1. SOCKET timeouts on the Hub client (grabette.hf.install_hub_timeouts).
#    huggingface_hub ships its httpx client with timeout=None, so a link that
#    DEGRADES rather than drops leaves a request waiting on a socket that never
#    delivers — forever, inside a thread nothing can interrupt. With per-operation
#    timeouts a stall raises inside the worker, hf_hub retries it, and the thread
#    DIES, leaving nothing behind. Covers everything that goes through httpx: the
#    JSON API and LFS uploads. It does NOT cover a xet upload, which is the
#    default path and does its HTTP in Rust — hence layer 3.
# 2. Retry with backoff here, so one bad minute doesn't sink a build that other
#    devices already spent twenty minutes uploading for.
# 3. A no-PROGRESS watchdog here, for a stall no socket timeout can see: a wedged
#    filesystem, or — the case that actually matters — a xet upload, whose bytes
#    move in a Rust HTTP stack that layer 1 does not touch at all. It lets the
#    COMMAND report back, so the fleet frees the device; the worker thread cannot
#    be killed, so it runs on.
#
# Layer 3 bounds SILENCE, never elapsed time. A total-elapsed cap cannot express
# the requirement, because any value it takes encodes an assumed throughput, and
# a healthy upload on a slower link then dies at the deadline — which is the one
# outcome we must never produce: an episode killed while it was still uploading
# fine. "No progress for N minutes" is the same guarantee against hangs with no
# assumption about speed, so a progressing upload runs as long as it needs to.
# The progress signal comes from grabette.hf's heartbeat (xet's own byte-level
# callback, plus httpx request/response hooks).
#
# Which is why uploads get their OWN executor. asyncio.to_thread uses the DEFAULT
# one, shared with the daemon's 50 Hz sensor poll, the OAK-D bring-up and
# stop_recording's mux — on a 4-core Pi that pool is 8 threads wide. An abandoned
# upload burning one of them competes for CPU with the H.264 encoders and the
# OAK-D drainers during a recording (hf_xet hashes and chunks in native threads),
# and enough of them would starve the capture path itself. Trading frame
# stability for upload robustness is not a trade worth making, so the two never
# share a pool: a stalled upload can only ever degrade uploading.
_UPLOAD_STALL_TIMEOUT_S = 900.0     # no Hub I/O at all for this long = wedged
_UPLOAD_WATCHDOG_POLL_S = 5.0       # how often staleness is judged. Costs nothing:
                                    # the wait returns as soon as the upload does,
                                    # so this only paces the CHECK, not the finish.
# Used ONLY when grabette.hf reports an INCOMPLETE heartbeat — some upload path
# is unobservable, so silence there proves nothing and the watchdog above must not
# rule on it. Partial coverage counts as incomplete on purpose: with only the
# httpx source wired, a xet transfer emits no marks at all and would read as a
# stall. Falling back to elapsed time accepts the false positive this whole design
# exists to avoid, because the alternative — no bound at all — is the original
# bug: a device pinned as "uploading" until someone restarts the Space.
# Deliberately far more generous than the 1800s it replaces, being a fallback now
# rather than the mechanism.
_UPLOAD_BLIND_ATTEMPT_TIMEOUT_S = 14400.0  # 4 h
_UPLOAD_MAX_ATTEMPTS = 3
_UPLOAD_RETRY_BASE_S = 5.0          # backoff: 5s, 10s (doubling per attempt)
# Deliberately narrow: uploads are sequential within a command, so one worker
# does the work and the second is headroom for a single abandoned thread.
_UPLOAD_WORKERS = 2

_upload_executor: "concurrent.futures.ThreadPoolExecutor | None" = None
_upload_executor_lock = threading.Lock()
# Threads we stopped waiting on that are still running. Self-correcting: each is
# counted down by its future's done callback if it ever finishes. Incremented on
# the event loop and decremented in a worker thread, hence the lock.
_stalled_uploads = 0
_stall_lock = threading.Lock()


def _mark_stalled(delta: int) -> int:
    """Adjust the stalled-upload count under the lock; returns the new value."""
    global _stalled_uploads
    with _stall_lock:
        _stalled_uploads = max(0, _stalled_uploads + delta)
        return _stalled_uploads


def _get_upload_executor():
    """The upload-only thread pool, created on first use.

    Lazily, not at import: a device that never runs a fleet dataset build should
    not carry the threads."""
    global _upload_executor
    with _upload_executor_lock:
        if _upload_executor is None:
            _upload_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=_UPLOAD_WORKERS, thread_name_prefix="hf-upload")
    return _upload_executor

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


def _space_excluded_episodes(quality) -> list[dict]:
    """The episodes the Space left out, one entry each, with its reason.

    The summary string above answers "why did this fail"; this answers "WHICH of
    my takes are not in the dataset". They are different questions and the second
    is the one an operator asks after a build that succeeded — a dataset quietly
    assembled from 17 of 20 recordings is worse than one that names the 3.

    The Space labels a role-layout episode "20250101_120000/left" (its
    episode_label), so the recording id and the arm are split back apart here:
    the fleet keys everything by episode id, and "which arm" is what tells the
    operator which grabette to go and look at.
    """
    if not isinstance(quality, list):
        return []
    out = []
    for ep in quality:
        if not isinstance(ep, dict) or not ep.get("excluded"):
            continue
        name = str(ep.get("name") or "").strip()
        if not name:
            continue
        episode_id, _, role = name.partition("/")
        reasons = [str(r).strip() for r in (ep.get("errors") or []) if str(r).strip()]
        out.append({"episode_id": episode_id, "role": role,
                    "reason": reasons[0] if reasons else
                              f"excluded by the conversion ({ep.get('verdict') or 'no verdict'})"})
    return out


class _UploadStalled(Exception):
    """An upload stopped making progress and was abandoned."""


async def _await_upload(cf, dest: str, attempt: int) -> None:
    """Wait for one upload, abandoning it only once it goes SILENT.

    Never returns because time passed: the only thing that ends the wait short of
    completion is the absence of Hub I/O for _UPLOAD_STALL_TIMEOUT_S. An upload
    that is still moving bytes — however slowly, however large the episode — keeps
    the heartbeat fresh and is never abandoned. That is the guarantee this
    function exists to provide.

    Shielded, so a poll that expires does NOT wait for a cancellation the worker
    thread cannot honour — awaiting that is the very hang being bounded. Nothing
    here runs off the event loop for longer than a dict lookup: the polling only
    reads two values from grabette.hf. Raises _UploadStalled, or whatever the
    upload itself raised.
    """
    from grabette import hf as hf_module

    # Mark now, so the clock starts at the submit. Without this, an upload that
    # never reaches the network at all (a wedged filesystem) would be measured
    # against whatever unrelated Hub call happened to mark last, or against
    # nothing at all.
    hf_module.note_hub_activity()
    started = time.monotonic()
    fut = asyncio.wrap_future(cf)
    try:
        while True:
            try:
                await asyncio.wait_for(asyncio.shield(fut),
                                       timeout=_UPLOAD_WATCHDOG_POLL_S)
                return
            except asyncio.TimeoutError:
                pass  # just a poll tick, not a verdict
            # Asked every tick, not once up front. The heartbeat is installed
            # lazily by HuggingFaceClient._get_api — in the WORKER thread we are
            # waiting on — so on the process's first upload no source exists yet
            # when we first get here. Installing it from this coroutine instead
            # would import the hf_xet extension on the EVENT LOOP (~100ms on x86,
            # more on a Pi), stalling the daemon's poll for that long. Asking
            # late costs one poll interval and blocks nothing.
            if hf_module.heartbeat_is_complete():
                age = hf_module.hub_activity_age_s()
                if age is not None and age >= _UPLOAD_STALL_TIMEOUT_S:
                    raise _UploadStalled(
                        f"no Hub activity for {age:.0f}s (attempt "
                        f"{attempt}/{_UPLOAD_MAX_ATTEMPTS}) — the upload is "
                        f"wedged, not slow; a slow one keeps the heartbeat alive")
            else:
                elapsed = time.monotonic() - started
                if elapsed >= _UPLOAD_BLIND_ATTEMPT_TIMEOUT_S:
                    logger.error(
                        "no upload heartbeat source for %s, so the bound fell "
                        "back to elapsed time — this abandonment MAY be a "
                        "healthy upload", dest)
                    raise _UploadStalled(
                        f"no progress signal available and {elapsed / 3600:.1f}h "
                        f"elapsed (attempt {attempt}/{_UPLOAD_MAX_ATTEMPTS})")
    except _UploadStalled:
        # Nothing will await `fut` again. Consume whatever the abandoned thread
        # eventually raises, or asyncio logs it as never retrieved — noise that
        # would land in the operator's journal long after the real error.
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())
        raise


async def _upload_one_episode(hf, ep_dir, raw_repo: str, dest: str, private: bool,
                              is_cancelled) -> None:
    """Upload one episode dir, bounded and retried. Raises on final failure.

    Two distinct failures are handled here, and conflating them is what made a
    flaky link fatal:
      • a transient error (reset connection, 5xx, stalled socket) — retried with
        a doubling backoff, because the alternative is losing every OTHER
        device's completed upload to one bad minute. Re-uploading the same folder
        is safe: an identical commit is a no-op on the Hub.
      • an attempt that STOPS MAKING PROGRESS — abandoned by _await_upload so the
        COMMAND can report back and the fleet can free this device. Note what is
        not in that sentence: how long the attempt has taken. A slow attempt is
        not a failed one, and killing one is the failure mode this whole layer is
        built to avoid. The worker thread cannot be killed (upload_folder is not
        interruptible), so it keeps running; it does so on the upload-only
        executor, where it can never delay a recording. See _UPLOAD_WORKERS.

    Runs on _get_upload_executor(), never asyncio.to_thread: to_thread means the
    DEFAULT executor, which the 50 Hz sensor poll and the OAK-D bring-up also use.

    Raises _CommandCancelled the moment the build is cancelled, including during
    a backoff wait: a cancel must not have to sit out a retry delay.
    """
    last: Exception | None = None
    for attempt in range(1, _UPLOAD_MAX_ATTEMPTS + 1):
        if is_cancelled():
            raise _CommandCancelled
        # Every worker tied up by a previous stall means this attempt would just
        # queue behind it — silently, for as long as the stall lasts. Say so
        # instead: it is a device-level fault with a device-level fix.
        if _stalled_uploads >= _UPLOAD_WORKERS:
            raise RuntimeError(
                f"{_stalled_uploads} earlier upload(s) are still stalled on this "
                "device and cannot be interrupted — reboot it before retrying")
        # submit() rather than loop.run_in_executor() so we keep the
        # concurrent.futures future: its done callback runs in the WORKER thread,
        # independent of the event loop. An asyncio future's callback is
        # scheduled ON the loop, so a loop that stops before the thread finishes
        # would leak the stall count permanently.
        cf = _get_upload_executor().submit(
            hf.upload_episode, ep_dir, raw_repo, None, dest, private)
        try:
            await _await_upload(cf, dest, attempt)
            return
        except _UploadStalled as e:
            last = e
            # Count the stall BEFORE registering the release, never after: a
            # thread that finishes in between would fire the callback first
            # (clamped to 0) and the increment would then leak a phantom stall
            # forever. add_done_callback on an already-finished future runs
            # inline, so the window is real, not theoretical.
            stalled = _mark_stalled(1)
            cf.add_done_callback(lambda _f: _mark_stalled(-1))
            logger.error("abandoning the upload thread for %s: %s (%d now stalled)",
                         dest, _exc_text(e), stalled)
        except Exception as e:  # noqa: BLE001 — retry whatever the Hub/network raised
            last = e
            logger.warning("upload of %s failed (attempt %d/%d): %s",
                           dest, attempt, _UPLOAD_MAX_ATTEMPTS, _exc_text(e))
        if attempt < _UPLOAD_MAX_ATTEMPTS:
            delay = _UPLOAD_RETRY_BASE_S * (2 ** (attempt - 1))
            # Sleep in slices so a cancel lands within a second, not after the
            # whole backoff.
            waited = 0.0
            while waited < delay:
                if is_cancelled():
                    raise _CommandCancelled
                await asyncio.sleep(min(1.0, delay - waited))
                waited += 1.0
    raise RuntimeError(
        f"{_UPLOAD_MAX_ATTEMPTS} attempts failed, last: {_exc_text(last)}"
        if last is not None else f"{_UPLOAD_MAX_ATTEMPTS} attempts failed")


# Relay commands that make this device too busy to record, and what to call the
# state. The fleet infers these from its own command queue, but the DEVICE could
# not see them at all: a relay command creates no local job, so the heartbeat
# reported "idle" all the way through a 20-minute upload. That blind spot is why
# the physical button could start a recording on top of one.
_DATASET_WORK_KINDS = {"upload_episodes": "uploading", "process_dataset": "processing"}

# command id -> kind, for the commands currently executing. The relay's worker is
# serial so this holds one entry at a time, but it is keyed by id rather than a
# flag so a lost entry can never wedge the device as permanently busy.
_active_dataset_work: dict[str, str] = {}


async def _handle_relay_command(cmd: dict) -> dict:
    """Relay entry point: record what this device is busy with, then dispatch.

    Tracked HERE rather than inside each handler so the bookkeeping cannot drift
    from the dispatch: every command type that occupies the device is declared in
    one table, and the finally covers every early return in the handlers below.
    """
    kind = _DATASET_WORK_KINDS.get(cmd.get("type") or "")
    if kind is None:
        return await _dispatch_relay_command(cmd)
    cmd_id = cmd.get("id") or ""
    _active_dataset_work[cmd_id] = kind
    try:
        return await _dispatch_relay_command(cmd)
    finally:
        _active_dataset_work.pop(cmd_id, None)


def _device_activity() -> str:
    """What this device is doing: idle | capturing | uploading | processing.

    ONE answer, used by both consumers: the fleet heartbeat (so an operator sees
    the true state) and the local capture gate (so a recording can't start on top
    of it). Two sources that could disagree is how a device ends up called free
    by one and busy by the other.

    Best-effort and non-throwing — a bad read reports 'idle'. All in-memory, so
    it stays cheap on the heartbeat.
    """
    from grabette.daemon import DaemonState
    from grabette.jobs import JobStatus, get_job_manager

    # Fleet-dispatched dataset work. First because it is the heaviest and the
    # least visible: it creates no local job, so nothing else here would see it.
    kinds = set(_active_dataset_work.values())
    if "processing" in kinds:
        return "processing"
    if "uploading" in kinds:
        return "uploading"
    # An upload we stopped waiting on is still RUNNING (see _upload_one_episode):
    # the thread cannot be killed, and on a Pi it is hashing and chunking, not
    # idling on a socket. The device is not free just because we gave up on it.
    if _stalled_uploads:
        return "uploading"
    try:
        active = [j for j in get_job_manager().list_jobs()
                  if j.status in (JobStatus.PENDING, JobStatus.RUNNING)]
        if any(j.name.startswith("push:") for j in active):
            return "processing"  # SLAM convert + push to the dataset
        if any(j.name.startswith("upload:") for j in active):
            return "uploading"
    except Exception:  # noqa: BLE001
        pass
    try:
        d = get_daemon_instance()
        if d is not None and d.state == DaemonState.RUNNING and d.backend.is_capturing:
            return "capturing"
    except Exception:  # noqa: BLE001
        pass
    return "idle"


# Activities during which starting a recording would produce a degraded episode.
# Mirrors the fleet's _RECORDING_BLOCKERS — the two gates must agree, or the
# fleet refuses a start the device would have accepted (or worse, the reverse).
_RECORDING_BLOCKERS = {
    "uploading": "this grabette is uploading episodes to the Hub — recording now "
                 "would compete with it for CPU and drop frames. Wait for the "
                 "upload to finish, or cancel the dataset build.",
    "processing": "this grabette is running a dataset conversion — recording now "
                  "would compete with it for CPU and drop frames. Wait for it to "
                  "finish, or cancel the dataset build.",
}


def _busy_reason() -> str:
    """Why capture is refused right now for BUSY reasons ("" = free).

    Installed on the backend at startup (set_busy_probe) so every start path —
    button, dashboard, fleet — goes through it. 'capturing' is deliberately NOT
    a blocker here: start_capture has its own "Already capturing" error, and
    reporting it as busy would mask that clearer message."""
    return _RECORDING_BLOCKERS.get(_device_activity(), "")


async def _dispatch_relay_command(cmd: dict) -> dict:
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
        from grabette.episode_check import missing_files

        tm = get_task_manager()
        hf = get_hf_client()
        uploaded, missing = [], []
        # Episodes present on disk but lacking a required artifact. Uploading one
        # is pure waste: the Space rejects it, and in a bimanual build one arm
        # short drops the WHOLE recording. Screening them here costs a stat() per
        # file and turns "the dataset came out empty" into a named reason, per
        # episode, attached to this device.
        incomplete: list[dict] = []
        present = []
        for eid in episode_ids:
            ep_dir = tm.episode_dir(eid)
            if not ep_dir.exists():
                missing.append(eid)
                continue
            lacks = missing_files(ep_dir)
            if lacks:
                # The arm matters as much as the recording: it says which
                # grabette to go and look at, and it is what lets the fleet
                # recognise the SAME arm being reported again by the conversion
                # (which drops the whole recording once an arm is absent) instead
                # of listing one lost take twice.
                incomplete.append({"episode_id": eid, "role": role, "missing": lacks})
                continue
            present.append((eid, ep_dir))

        # Nothing left to send: fail NOW instead of uploading zero episodes and
        # letting the build discover it has nothing to convert half an hour later,
        # after every other device has finished pushing.
        if not present and incomplete:
            # The message says WHAT happened; `incomplete` says which episodes and
            # why, and the fleet renders that per episode. Naming the files here
            # too just made the operator read the same causes twice.
            return {"status": "error", "role": role,
                    "message": (f"none of this device's {len(incomplete)} "
                                f"episode(s) can be converted"),
                    "uploaded": uploaded, "missing": missing, "incomplete": incomplete}

        for eid, ep_dir in present:
            # Cancellation checkpoint, once per episode. Checked BEFORE the first
            # upload too: the cancel may have landed while this command was still
            # queued behind another one (the relay's worker is serial), in which
            # case nothing should be uploaded at all. Per-episode is the finest
            # granularity available — one episode is a single blocking
            # api.upload_folder inside hf.upload_episode, which cannot be
            # interrupted from outside (see _upload_one_episode for the bound we
            # put on it).
            if cancels.is_cancelled(cmd_id):
                return {"status": "cancelled", "role": role, "uploaded": uploaded,
                        "missing": missing, "incomplete": incomplete}
            try:
                await _upload_one_episode(hf, ep_dir, raw_repo, f"{eid}/{role}", private,
                                          lambda: cancels.is_cancelled(cmd_id))
                uploaded.append(eid)
            except _CommandCancelled:
                return {"status": "cancelled", "role": role, "uploaded": uploaded,
                        "missing": missing, "incomplete": incomplete}
            except Exception as e:  # noqa: BLE001
                if cancels.is_cancelled(cmd_id):
                    # Cancelled mid-upload: the failure is a consequence, not news.
                    return {"status": "cancelled", "role": role, "uploaded": uploaded,
                            "missing": missing, "incomplete": incomplete}
                return {"status": "error", "message": f"upload failed for {eid}: {_exc_text(e)}",
                        "uploaded": uploaded, "missing": missing,
                        "incomplete": incomplete, "role": role}
        if cancels.is_cancelled(cmd_id):
            return {"status": "cancelled", "role": role, "uploaded": uploaded,
                    "missing": missing, "incomplete": incomplete}
        # Succeeded, but say what was left behind: a build quietly assembled from
        # 17 of 20 episodes is worse than one that says which 3 are gone and why.
        res = {"status": "ok", "role": role, "uploaded": uploaded,
               "missing": missing, "incomplete": incomplete}
        if incomplete:
            res["message"] = (f"uploaded {len(uploaded)} episode(s); skipped "
                              f"{len(incomplete)} that cannot be converted")
        return res

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
        # A SLAM check, not a dataset build: the run exists FOR its per-episode
        # tracking report, and a clean one deliberately pushes nothing. Two things
        # depend on knowing that — the Space's push decision, and the "no dataset"
        # guard below — so the fleet states it rather than letting either side infer
        # it from the target repo's name.
        check_only = bool(args.get("check_only", False))
        payload = {
            "source_repo": source_repo, "target_repo": target_repo,
            "task": args.get("task") or target_repo.split("/")[-1],
            "roles": args.get("roles") or [], "private": bool(args.get("private", False)),
            # Leave the raw dataset alone after the conversion (fleet-controlled,
            # default True there). Passed through untouched so the Space's own
            # default never silently deletes a raw we were asked to keep.
            "keep_raw": bool(args.get("keep_raw", True)),
            "check_only": check_only,
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
                        result_url = st.get("result")
                        excluded = _space_excluded_episodes(st.get("quality"))
                        if not result_url and check_only:
                            # A clean check pushes NOTHING by design, so "done" with
                            # no dataset is the GOOD outcome here — the answer is the
                            # report, not a repo. The fleet reads the report itself
                            # and words the verdict from it (including the case where
                            # every take was rejected, which also lands here), so this
                            # must not pre-empt it with a failure.
                            res = {"status": "ok"}
                            if excluded:
                                res["excluded"] = excluded
                            return res
                        if not result_url:
                            # "done" with no dataset is NOT a success. The Space
                            # finishes this way when every episode was rejected
                            # (bad recording check, no usable trajectory) — it
                            # logs the reason and pushes nothing. Reported as ok,
                            # it became a fleet job that invented a link to a repo
                            # that was never created, so the operator got a green
                            # "Open <dataset>" leading to a 404. Fail it here, and
                            # carry the Space's own reasons across.
                            # The per-episode reasons ride in `excluded`, which
                            # the fleet reports on its own. The pointer to the log
                            # is only for the case where the Space told us nothing
                            # structured at all — otherwise it would send the
                            # operator hunting for what is already on their screen.
                            return {"status": "error", "excluded": excluded, "message": (
                                "the conversion produced no dataset — no episode "
                                "made it through"
                                + ("" if excluded else " (see the Space log for details)"))}
                        # Succeeded, though episodes may still have been dropped:
                        # `excluded` names them, and that is the only place they
                        # need naming.
                        res = {"status": "ok", "result_url": result_url}
                        if excluded:
                            res["excluded"] = excluded
                        return res
                    if sstatus in ("error", "not_found"):
                        # not_found = the Space forgot the job. Its job list lives in
                        # memory, so this means it restarted mid-conversion (OOM, a
                        # redeploy) — say so, rather than echoing "not_found".
                        lost = (f"the Space no longer knows job {space_jid} — it restarted "
                                f"mid-conversion (its job list is kept in memory)")
                        detail = st.get("error") or (
                            lost if sstatus == "not_found" else "the Space reported a failure")
                        return {"status": "error", "message": detail,
                                "excluded": _space_excluded_episodes(st.get("quality"))}
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
    # The capture gate: every start path (button, dashboard, fleet) goes through
    # backend.raise_if_capture_blocked, which asks this. Installed BEFORE the
    # daemon starts and outside the relay block on purpose — a dashboard-driven
    # upload must block a recording just as a fleet-driven one does, and a device
    # with the relay disabled still runs those.
    backend.set_busy_probe(_busy_reason)
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
            # Hardware faults (no OAK-D calibration, no angle sensors) — the
            # device refuses to record in these states, and an operator who can
            # only find that out by walking up to the grabette and reading its
            # LED will find it out too late.
            fault_provider=lambda: getattr(_daemon.backend, "hardware_error", ""),
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
            from grabette.ui.app import APP_CSS, create_ui, fleet_theme

            demo = create_ui()
            # `allowed_paths` whitelists directories from which Gradio's
            # `/gradio_api/file=<path>` handler will serve files to the
            # browser. Without this, the multi-GB episode archives written
            # to data_dir/.downloads/ pass silently through — the file
            # exists on disk but Gradio refuses to hand it out, and the UI
            # never surfaces a download link. Default allowed paths cover
            # only Gradio's own cache and the OS temp dir.
            # theme/css belong HERE, not on gr.Blocks: Gradio 6 moved both to
            # launch()/mount_gradio_app, and a Blocks that is passed them just
            # warns and drops them on the floor.
            app = gr.mount_gradio_app(
                app, demo, path="/",
                theme=fleet_theme(),
                css=APP_CSS,
                # No framework footer at all:
                #  - "settings" is a second display-theme picker that does not
                #    write the preference our header switch stores, so the two
                #    silently contradict each other.
                #  - "api" opens Gradio's API recorder, which on this app just
                #    streams the dashboard's own 3 s polling timers back at the
                #    reader. The device API is documented in the repo instead.
                #  - "gradio" is the credit, and it is translated from
                #    navigator.language with no server-side override — a French
                #    browser got "Créé avec Gradio". Dropped rather than
                #    reproduced; Gradio is Apache-2.0 and asks for no in-app
                #    attribution.
                footer_links=[],
                allowed_paths=[str(settings.data_dir / ".downloads")],
            )
            logger.info("Gradio UI mounted at /")
        except ImportError:
            logger.warning(
                "Gradio not installed, UI disabled "
                "(install with: uv sync --extra ui)"
            )

    return app
