"""Entry point for `python -m gripette.bluetooth`.

Starts the BLE WiFi configuration service.
PIN is read from GRIPPER_BT_PIN env var (default: 00000).
"""

import logging
import os

from .bluetooth_service import BluetoothWifiService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

pin = os.environ.get("GRIPPER_BT_PIN", "00000")
service = BluetoothWifiService(device_name="Gripette", pin_code=pin)
service.run()
