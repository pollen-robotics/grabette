"""Hotspot fallback manager for Grabette.

Runs as root (systemd service). At startup:
  1. Waits up to WAIT_SECONDS for NetworkManager to connect to a home WiFi.
  2. If none connects → activates the grabette hotspot.

Then polls every POLL_SECONDS:
  - If home WiFi appears   → deactivates hotspot.
  - If connection is lost  → reactivates hotspot.
"""

from __future__ import annotations

import logging
import time

from grabette.config import settings
from grabette.wifi import (
    activate_hotspot,
    deactivate_hotspot,
    ensure_hotspot_profile,
    get_network_mode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WAIT_SECONDS = 30
POLL_SECONDS = 15


def main() -> None:
    logger.info(
        "Hotspot manager starting — will wait %ds for home WiFi (SSID: %s)",
        WAIT_SECONDS,
        settings.hotspot_ssid,
    )

    ensure_hotspot_profile(settings.hotspot_ssid, settings.hotspot_password)

    # Wait for home WiFi
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        if get_network_mode() == "connected":
            logger.info("Home WiFi connected at startup — hotspot standby")
            break
        time.sleep(2)
    else:
        logger.info("No home WiFi after %ds — activating hotspot", WAIT_SECONDS)
        activate_hotspot()

    # Monitor loop
    while True:
        time.sleep(POLL_SECONDS)
        mode = get_network_mode()
        if mode == "connected":
            logger.info("Home WiFi detected — deactivating hotspot")
            deactivate_hotspot()
        elif mode == "offline":
            logger.info("Network lost — activating hotspot")
            activate_hotspot()
        # mode == "hotspot": nothing to do


if __name__ == "__main__":
    main()
