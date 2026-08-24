"""HuggingFace Hub integration for episode upload and cloud SLAM."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# --- socket timeouts on every Hub call ----------------------------------------
# huggingface_hub builds its httpx client with timeout=None — no connect, read or
# write deadline at any layer. A link that DEGRADES rather than drops therefore
# leaves an upload waiting on a socket that never delivers, forever, inside a
# thread nothing can interrupt. That is the whole reason a grabette could stay
# pinned as "uploading" until the fleet Space was restarted.
#
# These are per-SOCKET-OPERATION budgets, not deadlines for the whole transfer:
# httpx counts them between chunks, so a legitimately slow multi-gigabyte upload
# over a weak link keeps running as long as it keeps making progress, and only a
# genuine stall trips them. When one does trip, it surfaces as
# httpx.TimeoutException INSIDE the worker thread — which hf_hub's own
# http_backoff already retries on, and which lets the thread DIE instead of
# lingering. That distinction is the point: an abandoned upload thread is far
# worse on a Pi than a slow one (see _UPLOAD_WORKERS in app/main.py).
#
# Scope, and it is narrower than it looks: this covers what goes through httpx,
# i.e. the JSON API and LFS uploads. A xet upload — the DEFAULT path, since
# hf-xet is a hard dependency of huggingface_hub on aarch64 — moves its bytes in
# a Rust stack that never touches this client, so none of these timeouts apply to
# it. The heartbeat below is what watches that path.
_HUB_CONNECT_TIMEOUT_S = 20.0
_HUB_READ_TIMEOUT_S = 60.0
_HUB_WRITE_TIMEOUT_S = 60.0   # no progress writing a chunk for this long = stalled
_HUB_POOL_TIMEOUT_S = 20.0

_client_factory_lock = threading.Lock()
_client_factory_installed = False


# --- upload liveness heartbeat -------------------------------------------------
# A total-elapsed bound on an upload cannot express what we need: whatever value
# it takes, it encodes an assumed throughput, and a healthy upload on a slower
# link gets killed. What we actually want to detect is SILENCE, so the bound has
# to be on "no progress since", which a progressing upload can never trip however
# long it legitimately runs.
#
# That needs a progress signal, and it needs one per upload path, because the two
# paths do their I/O in different stacks and NEITHER source sees the other's
# bytes:
#   - xet (the default: hf-xet is a hard dependency of huggingface_hub on every
#     machine we run on) does its HTTP in Rust, entirely OUTSIDE the httpx client
#     below. The socket timeouts never see a single byte of it. Hooked at
#     hf_xet.upload_files, which reports byte-level progress across both phases
#     (local hashing, then transfer).
#   - LFS and the plain JSON API go through get_session(), i.e. our httpx client,
#     where request/response hooks mark each call. Intra-request silence there is
#     already bounded by _HUB_READ/WRITE_TIMEOUT_S, so per-request marks suffice.
#
# Which path a given upload takes is decided SERVER-side (is this repo
# xet-enabled), so we cannot know it in advance — hence heartbeat_is_complete()
# demanding both before staleness may be trusted at all.
#
# ONE phase is observed by neither, and it is a deliberate accept rather than an
# oversight: huggingface_hub sha256s every file locally in
# CommitOperationAdd.__post_init__, before any network call, and nothing marks
# during it. It is bounded by an argument rather than by a timeout — capture
# itself sustains ~8 MB/s of writes for the whole recording (H.264 + lossless
# FFV1 depth), so a card that could record an episode can re-read it at least
# that fast. Hashing 3 GB at 8 MB/s is ~375s against a 900s stall budget, and
# sequential reads beat concurrent multi-stream writes, so the real margin is
# wider. A device too slow to clear that bar could not have produced the episode.
#
# Read by app/main.py's upload watchdog. Deliberately process-wide rather than
# per-upload: the callback is invoked from Rust threads, so there is nothing to
# key it on. The consequence is that unrelated Hub traffic (a whoami, a SLAM push,
# an abandoned upload thread that resumes) can refresh the heartbeat and delay an
# abandonment. That is the harmless direction — every residual imprecision here
# makes us wait LONGER, never kill an upload that was fine.
_activity_lock = threading.Lock()
_last_activity = 0.0
_heartbeat_sources: set[str] = set()
_heartbeat_installed = False

# The xet progress callback's parameter name, verified against the real signature
# at install time rather than assumed — injecting into the wrong slot would pass a
# function where the Rust side expects a repo type.
_XET_PROGRESS_ARG = "progress_updater"

# Every path must be observable before "it has gone quiet" means anything. A
# PARTIAL heartbeat is worse than none: with only httpx wired, a xet transfer
# emits no marks for its whole duration, and the watchdog would read that healthy
# transfer as a stall and abandon it — the exact failure this design exists to
# prevent.
_REQUIRED_SOURCES = frozenset({"xet", "httpx"})


def note_hub_activity() -> None:
    """Mark "the Hub link just did something". Cheap: one lock, one clock read."""
    global _last_activity
    with _activity_lock:
        _last_activity = time.monotonic()


def hub_activity_age_s() -> float | None:
    """Seconds since the last observed Hub I/O, or None if never marked."""
    with _activity_lock:
        if _last_activity == 0.0:
            return None
        return time.monotonic() - _last_activity


def heartbeat_sources() -> frozenset[str]:
    """Which progress sources are actually wired."""
    with _activity_lock:
        return frozenset(_heartbeat_sources)


def heartbeat_is_complete() -> bool:
    """True when every upload path is observable, so silence can be trusted.

    False means a watchdog must NOT conclude anything from staleness — see
    _UPLOAD_BLIND_ATTEMPT_TIMEOUT_S in app/main.py.
    """
    return _REQUIRED_SOURCES <= heartbeat_sources()


def _add_source(name: str) -> None:
    with _activity_lock:
        _heartbeat_sources.add(name)


# Fields on hf_xet's total-progress payload that say how much moved since the
# last tick. The first covers local hashing/chunking, the second the transfer, so
# together they span both phases of an upload.
_XET_INCREMENTS = ("total_bytes_completion_increment",
                   "total_transfer_bytes_completion_increment")


def _progress_moved(args) -> bool:
    """True if this progress tick reports real advancement.

    Marking on the mere ARRIVAL of a tick would make the heartbeat mean "the
    library is still talking" rather than "bytes are still moving". hf_xet ticks
    on a timer — it reports a completion RATE, which needs one — so a wedged
    transfer keeps ticking, and a heartbeat fed by arrivals would call it alive
    forever. That is the original bug, reintroduced one layer up.

    An unrecognised payload marks anyway. The two mistakes are not equal: failing
    to spot a hang costs a device held until the blind bound, while wrongly
    withholding a mark kills an upload that was fine. Only the first is
    acceptable.
    """
    total = args[0] if args else None
    known = [v for v in (getattr(total, n, None) for n in _XET_INCREMENTS)
             if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not known:
        return True
    return any(v > 0 for v in known)


def _marking_updater(inner):
    """A xet progress callback that marks the heartbeat, then defers to `inner`.

    `inner` is None whenever HF progress bars are off, because huggingface_hub
    then builds no reporter and passes None — still passing the argument. Chaining
    onto None is therefore what makes this work with progress bars disabled, which
    is the difference between a heartbeat that survives a quiet journald and one
    that does not.
    """
    def updater(*args, **kwargs):
        if _progress_moved(args):
            note_hub_activity()
        if inner is not None:
            return inner(*args, **kwargs)
        return None
    return updater


def _xet_progress_index(fn) -> int | None:
    """Position of the progress callback in fn's signature, or None if unknown."""
    import inspect

    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return None
    return params.index(_XET_PROGRESS_ARG) if _XET_PROGRESS_ARG in params else None


def _wrap_xet_upload(fn, index: int):
    """fn with the heartbeat chained onto whatever progress callback it is given."""
    def call(*args, **kwargs):
        if len(args) > index:
            args = args[:index] + (_marking_updater(args[index]),) + args[index + 1:]
        elif _XET_PROGRESS_ARG in kwargs:
            kwargs[_XET_PROGRESS_ARG] = _marking_updater(kwargs[_XET_PROGRESS_ARG])
        else:
            # Called in a shape we do not recognise. Uploading matters more than
            # observing it, so pass it straight through.
            logger.warning("xet upload called without a progress argument — "
                           "this upload is not observable")
        return fn(*args, **kwargs)

    call._grabette_heartbeat = True
    return call


def _install_xet_heartbeat() -> bool:
    """Mark the heartbeat from hf_xet's progress callback. True if wired.

    Hooked at hf_xet rather than at huggingface_hub's XetProgressReporter, even
    though the reporter looks like the more natural seam. Two reasons, both about
    not going blind: the reporter is only CONSTRUCTED when progress bars are
    enabled, so wrapping it ties the watchdog to a display setting anyone could
    switch off; and huggingface_hub imports these two functions inside its upload
    function, so replacing the attributes here is picked up on the next upload.
    """
    try:
        import hf_xet
    except Exception as e:  # noqa: BLE001
        logger.warning("No xet upload heartbeat (%s)", e)
        return False
    wired = False
    for name in ("upload_files", "upload_bytes"):
        fn = getattr(hf_xet, name, None)
        if fn is None:
            logger.warning("No xet heartbeat on hf_xet.%s: it is gone", name)
            continue
        if getattr(fn, "_grabette_heartbeat", False):
            wired = True
            continue
        index = _xet_progress_index(fn)
        if index is None:
            logger.warning("No xet heartbeat on hf_xet.%s: no %r in its signature",
                           name, _XET_PROGRESS_ARG)
            continue
        setattr(hf_xet, name, _wrap_xet_upload(fn, index))
        wired = True
    return wired


def install_upload_heartbeat() -> None:
    """Wire every available progress source. Idempotent, best effort."""
    global _heartbeat_installed
    if _heartbeat_installed:
        return
    if _install_xet_heartbeat():
        _add_source("xet")
    _heartbeat_installed = True
    if heartbeat_is_complete():
        logger.info("Upload heartbeat complete (%s)",
                    ", ".join(sorted(heartbeat_sources())))
    else:
        # Not cosmetic: this is the line that explains, after the fact, why an
        # upload was bounded by elapsed time instead of by silence.
        logger.error("Upload heartbeat INCOMPLETE — have {%s}, need {%s}. Upload "
                     "stalls can only be caught by the blind fallback bound.",
                     ", ".join(sorted(heartbeat_sources())) or "nothing",
                     ", ".join(sorted(_REQUIRED_SOURCES)))


def install_hub_timeouts() -> None:
    """Give every huggingface_hub call a socket timeout. Idempotent, best effort.

    set_client_factory is process-wide and public API. Called from _get_api so it
    covers every Hub user on the device (episode upload, SLAM push, whoami)
    without each of them having to remember."""
    global _client_factory_installed
    with _client_factory_lock:
        if _client_factory_installed:
            return
        try:
            import httpx
            from huggingface_hub import set_client_factory
            from huggingface_hub.utils._http import hf_request_event_hook

            timeout = httpx.Timeout(
                connect=_HUB_CONNECT_TIMEOUT_S, read=_HUB_READ_TIMEOUT_S,
                write=_HUB_WRITE_TIMEOUT_S, pool=_HUB_POOL_TIMEOUT_S,
            )
            # Marking on BOTH request and response, not just one: a request hook
            # fires before a call that may then block for its whole read timeout,
            # so on its own it would let a 60s stall look like fresh activity for
            # those 60s. The pair brackets the call instead.
            def _mark(_r) -> None:
                note_hub_activity()

            set_client_factory(lambda: httpx.Client(
                event_hooks={"request": [hf_request_event_hook, _mark],
                             "response": [_mark]},
                follow_redirects=True, timeout=timeout,
            ))
            _add_source("httpx")
            logger.info("Hub client timeouts installed (connect=%.0fs read=%.0fs "
                        "write=%.0fs)", _HUB_CONNECT_TIMEOUT_S, _HUB_READ_TIMEOUT_S,
                        _HUB_WRITE_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — never block Hub access over this
            # A future hf_hub could drop set_client_factory or move the event
            # hook. Losing the timeouts is a regression, not an outage: uploads
            # still work, they are just unbounded again (and app/main.py's own
            # attempt bound remains the backstop).
            logger.warning("Could not install Hub client timeouts: %s", e)
        _client_factory_installed = True


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub."""

    def __init__(self) -> None:
        self._api = None
        self._cached_token: str | None = None

    def set_token(self, token: str) -> None:
        """Persist token to the standard HF token file (or clear it)."""
        self._api = None
        self._cached_token = None
        if token:
            from huggingface_hub.constants import HF_TOKEN_PATH

            token_path = Path(HF_TOKEN_PATH)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(token)
        else:
            try:
                from huggingface_hub import logout

                logout()
            except Exception:  # noqa: BLE001
                pass

    @property
    def is_authenticated(self) -> bool:
        from huggingface_hub import get_token

        if not get_token():
            return False
        try:
            self._get_api()
            return True
        except Exception:
            return False

    def _get_api(self):
        from huggingface_hub import HfApi, get_token

        install_hub_timeouts()
        install_upload_heartbeat()
        token = get_token()
        if token != self._cached_token:
            self._api = None
            self._cached_token = token
        if self._api is None:
            if not token:
                raise ValueError("No token available")
            api = HfApi(token=token)
            api.whoami()
            self._api = api
        return self._api

    def get_user_info(self) -> dict | None:
        try:
            api = self._get_api()
            info = api.whoami()
            username = info.get("name", "")
            orgs = [o["name"] for o in info.get("orgs", []) if o.get("name")]
            namespaces = [username] + orgs if username else orgs
            return {"username": username, "email": info.get("email", ""), "namespaces": namespaces}
        except Exception:
            return None

    def upload_episode(
        self,
        episode_dir: Path,
        repo_id: str,
        progress_callback=None,
        path_in_repo: str | None = None,
        private: bool = False,
    ) -> str:
        """Upload an episode directory to HuggingFace Hub.

        Args:
            episode_dir: Path to the episode directory (the entire folder is uploaded)
            repo_id: HuggingFace repo ID (e.g., "username/grabette-data")
            progress_callback: Optional callable(percent: float, message: str)
            path_in_repo: Destination path inside the repo. Defaults to the
                episode id (the dir name). For a multi-device raw dataset pass
                "{episode_id}/{role}" so each device's stream for the SAME
                episode lands in its own subfolder instead of colliding.
            private: Whether the repository should be created as private

        Returns:
            URL of the uploaded data on HuggingFace Hub.
        """
        api = self._get_api()
        dest = path_in_repo or episode_dir.name

        if progress_callback:
            progress_callback(0.0, "Creating repository...")

        # Create repo if it doesn't exist. `private` takes effect only on the
        # FIRST create (exist_ok=True won't flip an existing repo's visibility);
        # since all of a job's devices pass the same value, whichever creates it
        # first sets it correctly.
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

        if progress_callback:
            progress_callback(10.0, "Uploading files...")

        # Upload the episode directory
        api.upload_folder(
            folder_path=str(episode_dir),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=dest,
        )

        if progress_callback:
            progress_callback(100.0, "Upload complete")

        url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{dest}"
        logger.info("Episode %s uploaded to %s", episode_dir.name, url)
        return url
