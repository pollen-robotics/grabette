"""Entry point for `python -m casquette.bluetooth`.

Starts the BLE WiFi configuration service.
PIN is read from CASQUETTE_BT_PIN env var (default: 00000).
"""

import logging
import os

from .bluetooth_service import BluetoothWifiService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

pin = os.environ.get("CASQUETTE_BT_PIN", "00000")
service = BluetoothWifiService(device_name="Casquette", pin_code=pin)
service.run()
