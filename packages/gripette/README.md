# gripette

Gripper version of the [Grabette](https://github.com/SteveNguyen/grabette) data collection system.
gRPC motor+camera service for the gripper, running on a Raspberry Pi Zero 2W.

Streams camera frames (JPEG) at ~10Hz synchronized with motor positions, and accepts motor commands for two Feetech STS3215 servos over the network.

## Hardware

- Raspberry Pi Zero 2W
- RPi camera module (1296x972, fisheye lens)
- Two Feetech STS3215 servos on `/dev/serial0` (baudrate 1000000, IDs 1 and 2)

## Installation

### Development machine (mock mode, no hardware needed)

```bash
uv sync --extra dev
uv run python generate_proto.py   # only needed if you modify gripper.proto
uv run python main.py
```

### Raspberry Pi Zero 2W

Requires system Python 3.11 with `libcamera` and `numpy` installed at the system level.

#### 1. UART configuration

The Pi Zero 2W has two UARTs: PL011 (`ttyAMA0`, full-featured, reliable at 1Mbaud) and mini UART (`ttyS0`, clock-dependent, unreliable at high baudrates). By default Bluetooth uses PL011 and GPIO gets the mini UART — this must be swapped for reliable servo communication.

Edit `/boot/firmware/config.txt` and add (or copy from `config/config.txt`):

```ini
dtoverlay=miniuart-bt
enable_uart=1
```

Edit `/boot/firmware/cmdline.txt` and **remove** `console=serial0,115200` (the kernel serial console conflicts with the servo bus). The file must remain a **single line** — do not add line breaks or the Pi will fail to boot.

Before:
```
console=serial0,115200 console=tty1 root=PARTUUID=... rootfstype=ext4 ...
```

After:
```
console=tty1 root=PARTUUID=... rootfstype=ext4 ...
```

Reboot, then verify:
```bash
ls -l /dev/serial0   # should point to ttyAMA0
```

#### 2. User permissions

Add your user to the `dialout` group for serial port access:

```bash
sudo usermod -aG dialout rasp
```

Log out and back in (or reboot) for the group change to take effect.

#### 3. Install dependencies

```bash
sudo apt install libcap-dev

uv venv --python /usr/bin/python3 --system-site-packages
uv sync --extra rpi --no-install-package numpy
uv run python main.py
```

## Configuration

All settings via environment variables with `GRIPPER_` prefix:

| Variable | Default | Description |
|---|---|---|
| `GRIPPER_HOST` | `0.0.0.0` | Server bind address |
| `GRIPPER_PORT` | `50051` | gRPC port |
| `GRIPPER_MOTOR_PORT` | `/dev/serial0` | Serial port for servos |
| `GRIPPER_MOTOR_BAUDRATE` | `1000000` | Serial baudrate |
| `GRIPPER_MOTOR_ID_1` | `1` | First servo ID |
| `GRIPPER_MOTOR_ID_2` | `2` | Second servo ID |
| `GRIPPER_JPEG_QUALITY` | `70` | JPEG compression quality |
| `GRIPPER_LOG_LEVEL` | `INFO` | Logging level |

## Usage

### Python client

```python
from gripette.client import GripperClient

with GripperClient("192.168.1.36:50051") as g:
    print(g.ping())

    g.torque_on()
    g.move(-0.5, -1.0)          # goal positions in radians

    for frame in g.stream():    # 10Hz camera + motor state
        print(f"Frame {frame.sequence}: {len(frame.jpeg_data)}B, "
              f"motors=({frame.motor1:.2f}, {frame.motor2:.2f})")
        break

    m1, m2 = g.read_motors()    # lightweight, no camera
    g.torque_off()
```

### Teleoperation bridge

Reads angle sensors from the grabette glove (Pi 4) and forwards them as motor commands to the gripper:

```bash
uv run python scripts/teleop_bridge.py --dry-run   # preview without moving motors
uv run python scripts/teleop_bridge.py              # live control
```

Requires the grabette service running on `192.168.1.35:8000`.

### Motor test

Sends a 1Hz sinusoidal command and records feedback positions for delay analysis:

```bash
uv run python scripts/sinus_test.py
# Outputs sinus_test.csv and sinus_test.png
```

### Camera test

Measures stream framerate and saves a sample frame:

```bash
uv run python scripts/camera_test.py
# Outputs camera_test.jpg
```

## systemd services

### Main service (gRPC motor+camera)

```bash
sudo cp systemd/gripette.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gripette
sudo systemctl start gripette
```

### Bluetooth WiFi configuration

A standalone BLE GATT service allows configuring WiFi credentials on the enclosed Pi Zero 2W without SSH or a screen. A phone or laptop connects via Bluetooth Low Energy, authenticates with a PIN, and sends WiFi credentials.

```bash
sudo apt install python3-dbus python3-gi   # system deps (usually pre-installed)
sudo cp systemd/gripette-bluetooth.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gripette-bluetooth
sudo systemctl start gripette-bluetooth
```

Runs as root (required by BlueZ DBus GATT registration). PIN is configurable via `GRIPPER_BT_PIN` env var (default: `00000`).

**BLE commands** (written as UTF-8 to the COMMAND characteristic):

| Command | Response | Description |
|---|---|---|
| `PING` | `PONG` | Health check |
| `PIN_xxxxx` | `OK: Connected` / `ERROR: Incorrect PIN` | Authenticate (required before WIFI/WIFI_RESET) |
| `WIFI ssid password` | `OK: Connecting to <ssid>` / `ERROR: ...` | Connect to WiFi via nmcli |
| `WIFI_RESET` | `OK: WiFi connections cleared` | Delete all saved WiFi networks |

Network status is also readable from a dedicated BLE characteristic (auto-updates every 10s).

**Web Bluetooth client**: open the [Gripette BT Tool](https://pollen-robotics.github.io/gripette/) in Chrome/Edge on a phone or laptop (requires HTTPS — served via GitHub Pages from `docs/index.html`).

See [docs/bluetooth_setup.md](docs/bluetooth_setup.md) for the full setup guide (BlueZ configuration, troubleshooting, etc.).

## Proto definition

The gRPC service contract is defined in `proto/gripper.proto`. To regenerate the Python files after modifying it:

```bash
uv sync --extra dev
uv run python generate_proto.py
```

Generated files in `gripette/proto/` are committed to the repository.
