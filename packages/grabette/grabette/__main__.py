"""Entry point for `python -m grabette`."""

import uvicorn

from grabette.app.main import create_app
from grabette.config import settings

app = create_app()
uvicorn.run(app, host=settings.host, port=settings.port)
