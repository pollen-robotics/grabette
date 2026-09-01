# Gripette Bluetooth WiFi Setup Guide

The gripette (Pi Zero 2W) is enclosed in the gripper with no physical access. A Bluetooth Low Energy (BLE) service lets you configure WiFi credentials from a phone or laptop — no SSH or screen needed.

## Prerequisites

- Raspberry Pi Zero 2W running Debian Bookworm (or later)
- BlueZ Bluetooth stack (pre-installed on Raspberry Pi OS)
- NetworkManager (`nmcli`) for WiFi management
- Chrome or Edge browser on the client device (Web Bluetooth is not supported in Firefox or Safari)

## Pi Setup

### 1. Install system dependencies

These are usually pre-installed on Raspberry Pi OS:

```bash
sudo apt install python3-dbus python3-gi bluez
```

### 2. Configure BlueZ for BLE-only mode

By default, BlueZ enables classic Bluetooth (BR/EDR) alongside BLE. Classic mode advertises audio profiles (A2DP, Handsfree, etc.) and uses the system hostname as the device name, which overrides the BLE service name.

Since the gripette only needs BLE, disable classic Bluetooth:

```bash
sudo nano /etc/bluetooth/main.conf
```

Find the `[General]` section (or create it) and set:

```ini
[General]
ControllerMode = le
```

Then restart BlueZ:

```bash
sudo systemctl restart bluetooth
```

Verify with:

```bash
sudo bluetoothctl show
```

You should see no audio UUIDs (A/V Remote Control, Handsfree, Audio Source/Sink). The device will advertise only as a BLE peripheral with the name "Gripette".

### 3. Install the systemd service

```bash
sudo cp systemd/gripette-bluetooth.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gripette-bluetooth
sudo systemctl start gripette-bluetooth
```

Check it's running:

```bash
sudo systemctl status gripette-bluetooth
```

### 4. (Optional) Change the PIN

The default PIN is `00000`. To change it, edit the systemd service:

```bash
sudo systemctl edit gripette-bluetooth
```

Add:

```ini
[Service]
Environment=GRIPPER_BT_PIN=12345
```

Then restart:

```bash
sudo systemctl restart gripette-bluetooth
```

## Configuring WiFi from a Phone or Laptop

### 1. Open the Web Bluetooth tool

Open the [BT Tool](https://pollen-robotics.github.io/grabette/) in **Chrome** or **Edge**, then pick **Gripette** in the device chooser (Web Bluetooth requires HTTPS — Firefox and Safari are not supported). It's the single shared page that provisions any robot.

### 2. Connect

Click **"Connect to Gripette"**. A browser popup will show nearby BLE devices. Select **"Gripette"** and click Pair.

### 3. Verify connection

Click **"Ping"**. You should see `PONG` in the log.

### 4. Authenticate

Enter the PIN (default: `00000`) and click **"Authenticate"**. The log should show `OK: Connected`.

### 5. Send WiFi credentials

Enter the SSID and password, then click **"Connect WiFi"**. The Pi will attempt to connect via `nmcli`.

### 6. Verify

Click **"Read Network Status"**. If successful, you should see something like:

```
CONNECTED [wlan0] 192.168.1.36
```

The gripette is now on your WiFi network.

### 7. Set the hand

`gripette.service` will not start until it knows which hand this gripper is: on
every start `ExecStartPre` runs `hand-from-hostname`, which derives left/right
from the **hostname** and appends `GRIPPER_HAND` to `/etc/gripette/env`. On
GripetteOS the hostname used to be set while flashing, in Raspberry Pi Imager's
OS customisation — Imager 2.0 removed that for custom images, so the BT Tool
sets it instead.

The tool shows the current hand as soon as it connects (reading it needs no
PIN). Click **Set to left** or **Set to right** — that needs the PIN, like the
WiFi commands. It renames the host to `gripette-<hand>` — keeping the
per-device serial suffix GripetteOS appends on first boot, so `gripette-1f9a2b`
becomes `gripette-left-1f9a2b` — clears the cached
`GRIPPER_HAND` line so the derivation runs again, and restarts the service;
the `GRIPPER_MOTOR*_OFFSET` calibration in the same env file is left untouched
and stays valid (the hand only flips the motor signs, and the offsets are
stored in the encoder frame).

## BLE command reference

The web tool wraps these, but the commands are also usable directly — written as UTF-8 to the COMMAND characteristic; responses arrive as notifications:

| Command | Response | Description |
|---|---|---|
| `PING` | `PONG` | Health check |
| `PIN_xxxxx` | `OK: Connected` / `ERROR: Incorrect PIN` | Authenticate (required before WIFI/WIFI_RESET) |
| `WIFI ssid password` | `OK: Connecting to <ssid>` / `ERROR: ...` | Connect to WiFi via nmcli |
| `WIFI_RESET` | `OK: WiFi connections cleared` | Delete all saved WiFi networks |
| `HAND` | `{"hand":"left\|right\|null","hostname":"..."}` | Read the hand this gripette resolved (no PIN needed) |
| `HAND left\|right` | `OK: Hand set to <hand> (hostname gripette-<hand>)` / `ERROR: ...` | Set the hand (needs the PIN, consumes it) |

Network status is also readable from a dedicated BLE characteristic (auto-updates every 10s).

## Troubleshooting

### Device shows as hostname (e.g., "raspiZ") instead of "Gripette"

Classic Bluetooth is still active. Follow step 2 above to set `ControllerMode = le` in `/etc/bluetooth/main.conf`.

### Device appears as an audio device

Same cause — classic Bluetooth profiles are active. Set `ControllerMode = le`.

### "Incorrect PIN or passkey" when pairing

Stale pairing data. Remove the device from both sides:

- **On your phone/laptop**: Bluetooth settings → find the device → Forget/Remove
- **On the Pi**:
  ```bash
  sudo bluetoothctl
  paired-devices
  remove XX:XX:XX:XX:XX:XX   # use the MAC address shown
  exit
  sudo systemctl restart gripette-bluetooth
  ```

### Web page shows "Disconnected" and nothing happens on click

- **Wrong browser**: Web Bluetooth only works in Chrome and Edge. Firefox and Safari are not supported.
- **Not HTTPS**: Web Bluetooth requires a secure context. Use the GitHub Pages link (HTTPS) or `localhost` for local testing:
  ```bash
  python3 -m http.server 8080 --directory docs/
  # Then open http://localhost:8080 in Chrome
  ```

### No BLE devices found in the browser popup

- Make sure the bluetooth service is running: `sudo systemctl status gripette-bluetooth`
- Check that BLE is unblocked: `rfkill list bluetooth`
- If blocked: `sudo rfkill unblock bluetooth`

### WiFi connection fails

- Verify the SSID and password are correct
- Check that NetworkManager is running: `systemctl status NetworkManager`
- Check available networks: `nmcli device wifi list`
- Check logs: `sudo journalctl -u gripette-bluetooth -f`

### Service won't start

Check logs for the actual error:

```bash
sudo journalctl -u gripette-bluetooth -e
```

Common causes:
- Missing dependencies: `sudo apt install python3-dbus python3-gi`
- BlueZ not running: `sudo systemctl start bluetooth`
- Another process using the BLE adapter
