# Grabette — Bluetooth WiFi configuration

A standalone BLE GATT service (`grabette-bluetooth.service`) lets you configure WiFi credentials without SSH or a screen. It's installed + started by `make install-systemd` (see the [README](../README.md)). Once running:

Connect from a phone or laptop via Bluetooth Low Energy on the [BT Tool](https://pollen-robotics.github.io/grabette/) in Chrome/Edge and follow those steps:
1. Select Grabette and click on Connect
2. Select your Grabette on the pop-up, then Pair
3. Authenticate with the PIN
4. Scan networks, select your wifi and send WiFi credentials.

PIN is configurable via the `GRABETTE_BT_PIN` env var (default: `00000`); set it in `systemd/grabette-bluetooth.service` (`Environment=GRABETTE_BT_PIN=...`) before installing.

**Commands** (written to the COMMAND characteristic as UTF-8; responses arrive as notifications):

| Command | Response |
|---|---|
| `PING` | `PONG` |
| `PIN_xxxxx` | `OK: Connected` / `ERROR: Incorrect PIN` (required before the WIFI commands) |
| `WIFI_SCAN` | JSON array of nearby SSIDs (strongest first) |
| `WIFI ssid password` | `OK: Connecting to <ssid>` / `ERROR: ...` (connects via an explicit WPA-PSK profile) |
| `WIFI_RESET` | `OK: WiFi connections cleared` |

The adapter advertises with `Pairable = True` and uses the `NoInputNoOutput` agent for silent Just Works pairing — required because some centrals (notably Windows and some Linux/BlueZ stacks) refuse GATT operations until they've bonded. macOS clients can still connect "connection-only" without bonding; both modes work.

> **If a client gets stuck pairing** (e.g. a stale bond from an earlier version that used `Pairable = False`): clear it on both ends — `bluetoothctl remove <mac.address.of.Grabette>` on the Pi and the client, plus Forget the device in `chrome://bluetooth-internals`.
