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

# Fleet-driven scheduled start (synchronized group recordings). A group start
# fans out the same start_at_utc to every member's queue (see grabette-fleet's
# /api/fleet/groups/{id}/start_capture) — each device waits it out on its own
# NTP-disciplined clock, so the group starts in lockstep regardless of which
# device happens to poll first. Module-level because _handle_relay_command is
# a free function re-entered on every poll cycle; there's only ever one
# capture (scheduled or running) per device.
_scheduled_task: asyncio.Task | None = None
_scheduled_start_utc: datetime | None = None
# True once T0 has fired and backend.start_capture is in flight. Distinguishes
# "safe to cancel" (still waiting) from "must let it finish" (hardware init in
# progress) when a stop_capture races the scheduled start.
_scheduled_starting: bool = False


async def _wait_and_start(target_utc: datetime, episode_dir: Path, sm) -> None:
    global _scheduled_task, _scheduled_start_utc, _scheduled_starting
    try:
        wait_s = (target_utc - datetime.now(timezone.utc)).total_seconds()
        if wait_s > 0:
            await asyncio.sleep(wait_s)
        _scheduled_starting = True
        try:
            daemon = get_daemon_instance()
            await daemon.backend.start_capture(episode_dir)
            logger.info("Scheduled start fired (target %s)", target_utc.isoformat())
        finally:
            _scheduled_starting = False
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Scheduled start failed; discarding pending episode")
        sm.discard_pending_episode()
    finally:
        _scheduled_task = None
        _scheduled_start_utc = None


async def _handle_relay_command(cmd: dict) -> dict:
    """Map fleet commands to grabette daemon actions."""
    global _scheduled_task, _scheduled_start_utc, _scheduled_starting
    from grabette.daemon import DaemonState
    from grabette.app.routers.sessions import get_session_manager

    ctype = cmd.get("type")
    daemon = get_daemon_instance()
    if daemon is None:
        return {"status": "error", "message": "daemon not running"}

    if ctype == "get_state":
        status = daemon.status
        if _scheduled_task is not None and not _scheduled_task.done():
            status["scheduled_start_utc"] = _scheduled_start_utc.isoformat() if _scheduled_start_utc else None
        return {"status": "ok", "state": status}

    if ctype == "logout":
        from huggingface_hub import logout as hf_logout
        hf_logout()
        return {"status": "ok"}

    if daemon.state != DaemonState.RUNNING:
        return {"status": "error", "message": f"daemon not ready ({daemon.state.value})"}

    backend = daemon.backend
    if ctype == "start_capture":
        if backend.is_capturing:
            return {"status": "error", "message": "already capturing"}
        if _scheduled_task is not None and not _scheduled_task.done():
            return {"status": "error", "message": "a start is already scheduled"}
        sm = get_session_manager()
        args = cmd.get("args", {})
        task_name = args.get("task_name")
        session_id = sm.get_or_create_session(task_name) if task_name else args.get("session_id")
        start_at_utc = args.get("start_at_utc")

        episode_id = sm.create_episode(session_id)
        episode_dir = sm.episode_dir(episode_id)

        if not start_at_utc:
            try:
                await backend.start_capture(episode_dir)
            except Exception:
                sm.discard_pending_episode()
                raise
            return {"status": "ok", "episode_id": episode_id}

        # Scheduled (synchronized group) start: wait for T0 in the background
        # and ack immediately so the fleet dispatch round-trip doesn't block.
        try:
            target = datetime.fromisoformat(start_at_utc)
        except ValueError:
            sm.discard_pending_episode()
            return {"status": "error", "message": f"invalid start_at_utc: {start_at_utc!r}"}
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        if target <= datetime.now(timezone.utc):
            sm.discard_pending_episode()
            return {"status": "error", "message": "start_at_utc is in the past"}

        _scheduled_start_utc = target
        _scheduled_task = asyncio.create_task(
            _wait_and_start(target, episode_dir, sm), name=f"scheduled-start-{episode_id}",
        )
        return {"status": "scheduled", "episode_id": episode_id, "start_at_utc": target.isoformat()}

    if ctype == "stop_capture":
        if _scheduled_task is not None and not _scheduled_task.done():
            if _scheduled_starting:
                # T0 already fired, hardware init in flight — wait it out
                # rather than interrupt it, then fall through to a real stop.
                try:
                    await asyncio.wait_for(_scheduled_task, timeout=15.0)
                except asyncio.TimeoutError:
                    return {"status": "error", "message": "start_capture still running after 15s; refusing to stop"}
                if not backend.is_capturing:
                    return {"status": "cancelled"}
            else:
                _scheduled_task.cancel()
                try:
                    await _scheduled_task
                except asyncio.CancelledError:
                    pass
                get_session_manager().discard_pending_episode()
                return {"status": "cancelled"}
        if not backend.is_capturing:
            return {"status": "error", "message": "not capturing"}
        result = await backend.stop_capture()
        get_session_manager().register_episode(getattr(result, "session_id", None))
        return {"status": "ok", "result": result}
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
            from grabette.app.routers.sessions import get_session_manager

            _button_listener = ButtonListener(backend, get_session_manager())
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
    from grabette.app.routers.sessions import router as sessions_router
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
    app.include_router(sessions_router)
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
