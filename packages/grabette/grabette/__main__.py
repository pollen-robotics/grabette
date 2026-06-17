"""Entry point for `python -m grabette` and the `grabette` CLI command."""

import uvicorn

from grabette.app.main import create_app
from grabette.config import settings


def main() -> None:
    app = create_app()
    uvicorn.run(app, host=settings.host, port=settings.port)


main()
