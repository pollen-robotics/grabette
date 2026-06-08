from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from casquette.config import settings
from casquette.daemon import Daemon

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
        from casquette.backend.mock import MockBackend
        logger.info("Using MockBackend (forced by config)")
        return MockBackend()
    elif settings.backend == "rpi":
        from casquette.backend.rpi import RpiBackend
        logger.info("Using RpiBackend (forced by config)")
        return RpiBackend(imu_i2c_bus=settings.imu_i2c_bus)
    else:  # auto
        try:
            from casquette.backend.rpi import RpiBackend
            import picamera2  # noqa: F401
            logger.info("RPi hardware detected, using RpiBackend")
            return RpiBackend(imu_i2c_bus=settings.imu_i2c_bus)
        except ImportError:
            from casquette.backend.mock import MockBackend
            logger.info("No RPi hardware, using MockBackend")
            return MockBackend()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _daemon

    backend = _create_backend()
    _daemon = Daemon(backend)
    await _daemon.start()

    yield

    await _daemon.stop()
    _daemon = None


def create_app() -> FastAPI:
    from casquette.app.routers.camera import router as camera_router
    from casquette.app.routers.daemon import router as daemon_router
    from casquette.app.routers.sessions import router as sessions_router
    from casquette.app.routers.state import router as state_router
    from casquette.app.routers.system import router as system_router

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
    app.include_router(sessions_router)
    app.include_router(camera_router)
    app.include_router(system_router)

    return app
