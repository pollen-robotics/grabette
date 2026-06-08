"""Entry point for `python -m casquette`."""

import uvicorn

from casquette.app.main import create_app
from casquette.config import settings

app = create_app()
uvicorn.run(app, host=settings.host, port=settings.port)
