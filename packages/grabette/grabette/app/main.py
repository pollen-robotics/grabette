from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from grabette.config import settings
from grabette.daemon import Daemon

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


async def _handle_relay_command(cmd: dict) -> dict:
    """Map fleet commands to grabette daemon actions.

    start_capture/stop_capture share their scheduling state machine (see
    capture_scheduler.py) with the physical button and the local UI, so a
    fleet-dispatched synchronized start (start_at_utc set by a group's
    /api/fleet/groups/{id}/start_capture or another device's local trigger)
    behaves identically to one triggered locally on this device.
    """
    from grabette.capture_scheduler import get_capture_scheduler
    from grabette.daemon import DaemonState
    from grabette.task import episode_id_for
    from grabette.app.routers.tasks import get_task_manager

    ctype = cmd.get("type")
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

        episode_id = tm.create_episode(task_id, episode_id=episode_id_for(target) if target else None)
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

    # Start fleet relay loop
    relay_task = None
    if settings.relay_enabled:
        from huggingface_hub import get_token
        from grabette.relay_client import RelayClient

        relay = RelayClient(
            base_url=settings.relay_url,
            token_provider=get_token,
            device_id=settings.device_id,
            name=settings.device_name,
            capabilities=["get_state", "start_capture", "stop_capture", "logout"],
            hand=settings.hand,
        )
        relay_task = asyncio.create_task(relay.run(_handle_relay_command))
        logger.info("Relay started → %s (device: %s)", settings.relay_url, settings.device_id)

    yield

    if relay_task is not None:
        relay_task.cancel()
        import contextlib
        with contextlib.suppress(asyncio.CancelledError):
            await relay_task

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
    from grabette.auth import HFAuth
    from grabette.webauth import build_auth_router

    _hf_auth = HFAuth()
    app.include_router(build_auth_router(_hf_auth))

    # Mount Gradio UI if enabled and installed
    if settings.ui_enabled:
        try:
            import gradio as gr
            from grabette.ui.app import create_ui

            demo = create_ui()
            app = gr.mount_gradio_app(app, demo, path="/")
            logger.info("Gradio UI mounted at /")
        except ImportError:
            logger.warning(
                "Gradio not installed, UI disabled "
                "(install with: uv sync --extra ui)"
            )

    return app
