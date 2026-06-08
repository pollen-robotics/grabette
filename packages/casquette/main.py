"""Entry point for casquette service."""

import uvicorn

from casquette.app.main import create_app
from casquette.config import settings

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
