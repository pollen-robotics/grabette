"""Entry point for `python -m grabette`."""

import uvicorn

from grabette.app.main import create_app
from grabette.config import settings

app = create_app()
# A dashboard tab can hold a response that never finishes (a paused <video>
# stops reading its file): without a bound, a stop waits on it until systemd
# SIGKILLs the service 90 s later, skipping the camera teardown.
uvicorn.run(app, host=settings.host, port=settings.port,
            timeout_graceful_shutdown=3)
