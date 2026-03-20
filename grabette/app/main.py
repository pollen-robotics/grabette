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
_grpc_server = None


def get_daemon_instance() -> Daemon | None:
    return _daemon


def get_grpc_server_instance():
    return _grpc_server


def _create_backend():
    """Create backend based on config (auto-detect, mock, or rpi)."""
    if settings.backend == "mock":
        from grabette.backend.mock import MockBackend
        logger.info("Using MockBackend (forced by config)")
        return MockBackend()
    elif settings.backend == "rpi":
        from grabette.backend.rpi import RpiBackend
        logger.info("Using RpiBackend (forced by config)")
        return RpiBackend(enable_angle=settings.angle_sensors)
    else:  # auto
        try:
            from grabette.backend.rpi import RpiBackend
            import picamera2  # noqa: F401
            logger.info("RPi hardware detected, using RpiBackend")
            return RpiBackend(enable_angle=settings.angle_sensors)
        except ImportError:
            from grabette.backend.mock import MockBackend
            logger.info("No RPi hardware, using MockBackend")
            return MockBackend()


_button_listener = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _daemon, _button_listener, _grpc_server
    import asyncio

    backend = _create_backend()

    # Start gRPC server and wrap backend with recording hooks
    if settings.grpc_enabled:
        from grabette.grpc_server import GrpcServer
        from grabette.backend.grpc_wrapper import GrpcBackend
        _grpc_server = GrpcServer(host=settings.grpc_host, port=settings.grpc_port)
        if _grpc_server.start():
            backend = GrpcBackend(backend, _grpc_server)

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

    yield

    if _button_listener is not None:
        _button_listener.stop()
        _button_listener = None
    await _daemon.stop()
    _daemon = None
    if _grpc_server is not None:
        _grpc_server.stop()
        _grpc_server = None  # noqa: F841 (module-level reset)


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
    from grabette.app.routers.grpc import router as grpc_router
    from grabette.app.routers.replay import router as replay_router
    from grabette.app.routers.viewer import router as viewer_router

    app.include_router(daemon_router)
    app.include_router(state_router)
    app.include_router(sessions_router)
    app.include_router(camera_router)
    app.include_router(hf_router)
    app.include_router(system_router)
    app.include_router(viewer_router)
    app.include_router(charts_router)
    app.include_router(replay_router)
    app.include_router(grpc_router)

    # Serve URDF model + STL meshes as static files
    _urdf_dir = Path(__file__).resolve().parent.parent.parent / "urdf"
    if _urdf_dir.is_dir():
        app.mount("/urdf", StaticFiles(directory=str(_urdf_dir)), name="urdf")
        logger.info("URDF assets mounted at /urdf from %s", _urdf_dir)

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
