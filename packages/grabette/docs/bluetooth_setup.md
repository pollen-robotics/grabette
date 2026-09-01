# Grabette — Bluetooth WiFi configuration

A standalone BLE GATT service (`grabette-bluetooth.service`) lets you configure WiFi credentials without SSH or a screen. It's installed + started by `make install-systemd` (see the [README](../README.md)). Once running:

Connect from a phone or laptop via Bluetooth Low Energy on the [BT Tool](https://pollen-robotics.github.io/grabette/) in **Chrome/Edge** and follow those steps:
1. Select Grabette and click on Connect
2. Select your Grabette on the pop-up, then Pair
3. Authenticate with the PIN
4. Scan networks, select your wifi and send WiFi credentials.
5. Set the hand — **left** or **right**.

## Setting the hand

`grabette.service` will not start until it knows which hand this device is: on
every start `ExecStartPre` runs `hand-from-hostname`, which derives left/right
from the **hostname** and appends `GRABETTE_HAND` to `/etc/grabette/env`. On
GrabetteOS the hostname used to be set while flashing, in Raspberry Pi Imager's
OS customisation — Imager 2.0 removed that for custom images, so the BT Tool
sets it instead.

The tool shows the current hand as soon as it connects (reading it needs no
PIN), and the **Set to left** / **Set to right** buttons need the PIN like the
WiFi ones. Setting it renames the host to `grabette-<hand>`, clears the cached
`GRABETTE_HAND` line so the derivation runs again, and restarts the service —
per-device calibration in the same env file is left untouched. It is safe to
change later: switching hands only flips the sensor signs, and
`scripts/calibrate_angles.py` does not need re-running.

> Be careful, Chrome may need to enable experimental features : <code>chrome://flags/#enable-experimental-web-platform-features</code>

PIN is configurable via the `GRABETTE_BT_PIN` env var (default: `00000`); set it in `systemd/grabette-bluetooth.service` (`Environment=GRABETTE_BT_PIN=...`) before installing.

**Commands** (written to the COMMAND characteristic as UTF-8; responses arrive as notifications):

| Command | Response |
|---|---|
| `PING` | `PONG` |
| `PIN_xxxxx` | `OK: Connected` / `ERROR: Incorrect PIN` (required before the WIFI commands) |
| `WIFI_SCAN` | JSON array of nearby SSIDs (strongest first) |
| `WIFI ssid password` | `OK: Connecting to <ssid>` / `ERROR: ...` (connects via an explicit WPA-PSK profile) |
| `WIFI_RESET` | `OK: WiFi connections cleared` |
| `HAND` | `{"hand":"left\|right\|null","hostname":"..."}` (no PIN needed) |
| `HAND left\|right` | `OK: Hand set to <hand> (hostname grabette-<hand>)` / `ERROR: ...` (needs the PIN, consumes it) |

The adapter advertises with `Pairable = True` and uses the `NoInputNoOutput` agent for silent Just Works pairing — required because some centrals (notably Windows and some Linux/BlueZ stacks) refuse GATT operations until they've bonded. macOS clients can still connect "connection-only" without bonding; both modes work.

> **If a client gets stuck pairing** (e.g. a stale bond from an earlier version that used `Pairable = False`): clear it on both ends — `bluetoothctl remove <mac.address.of.Grabette>` on the Pi and the client, plus Forget the device in `chrome://bluetooth-internals`.
