"""Entry point for gripette gRPC service."""

import logging

from gripette.config import settings
from gripette.server import serve

if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    serve()
