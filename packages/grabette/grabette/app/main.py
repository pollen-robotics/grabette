from __future__ import annotations

import logging
from contextlib import asynccontextmanager
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


async def _handle_relay_command(cmd: dict) -> dict:
    """Map fleet commands to grabette daemon actions."""
    from grabette.daemon import DaemonState
    from grabette.app.routers.sessions import get_session_manager

    ctype = cmd.get("type")
    daemon = get_daemon_instance()
    if daemon is None:
        return {"status": "error", "message": "daemon not running"}

    if ctype == "get_state":
        return {"status": "ok", "state": daemon.status}

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
        sm = get_session_manager()
        session_id = cmd.get("args", {}).get("session_id")
        episode_id = sm.create_episode(session_id)
        episode_dir = sm.episode_dir(episode_id)
        try:
            await backend.start_capture(episode_dir)
        except Exception:
            sm.discard_pending_episode()
            raise
        return {"status": "ok", "episode_id": episode_id}
    if ctype == "stop_capture":
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
