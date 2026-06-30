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

Prerequisites: a Pi Zero 2W running Raspberry Pi OS (Bookworm or Trixie), with [uv](https://docs.astral.sh/uv/) installed.

```bash
sudo usermod -aG dialout $USER   # serial bus access — log out + back in for it to take effect
make install-rpi                  # one-shot: apt deps + UART config + venv + sync + verify
sudo reboot                       # required if the UART config was changed
make check                        # post-reboot hardware diagnostic (camera + motors)
```

`make check` validates the camera and the motor bus. It also probes the two systemd services and reports them as `[SKIP]` if they aren't installed yet — that's the expected state right after `install-rpi`.

Then start the service manually or install at boot:

```bash
uv run --package gripette python -m gripette   # foreground (Ctrl-C to stop)
# — or —
make install-systemd                            # boot-time start (main + bluetooth)
make check                                      # services should now report [OK]
```

`make install-rpi` is idempotent — re-running it is safe. Under the hood it:

- installs `python3-libcamera`, `python3-picamera2`, `libcap-dev` via apt;
- runs `make enable-uart` to disable the serial console (`cmdline.txt`) and add `dtoverlay=miniuart-bt` to `config.txt` so the reliable PL011 (`ttyAMA0`) ends up on the GPIO header instead of the mini UART (clock-dependent, unreliable at 1Mbaud);
- creates a `--system-site-packages` venv at the workspace root so apt's `picamera2` satisfies the dependency tree (otherwise `uv` tries to build `python-prctl` from PyPI);
- runs `uv sync --package gripette --extra rpi --no-install-package numpy` and verifies that `picamera2`, `serial`, and `rustypot` all import.

`make help` lists every target. The cmdline.txt edit captures the `root=PARTUUID=...` token before editing and rolls back from a `.gripette.bak` backup if it changes — boot is safe.

#### Manual installation (fallback)

If `make install-rpi` fails (e.g. unusual OS), the equivalent manual steps are:

1. **UART**: edit `/boot/firmware/config.txt` to include `dtoverlay=miniuart-bt` and `enable_uart=1`. Edit `/boot/firmware/cmdline.txt` to remove `console=serial0,115200` — keep the file as a single line. Reboot.
2. **Deps**: `sudo apt install libcap-dev python3-libcamera python3-picamera2`.
3. **Venv**: from the workspace root, `uv venv --python /usr/bin/python3 --system-site-packages && uv sync --package gripette --extra rpi --no-install-package numpy`.

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

### Motor assembly

A gripette uses two Feetech STS3215 servos with distinct IDs. Brand-new motors all ship as ID=1 at 1Mbaud in position mode, so for each new gripper one of the two motors must be reconfigured before assembly.

| role     | motor_id | physical position |
|----------|----------|-------------------|
| proximal | 1        | base of the finger |
| distal   | 2        | tip of the finger  |

Use `configure_motor.py` to set each motor's ID. Connect **one motor at a time** on the bus (two motors both at ID=1 collide and the bus returns nothing usable):

```bash
uv run python scripts/configure_motor.py             # interactive: prompts for role
uv run python scripts/configure_motor.py --info      # read-only: prints current config
uv run python scripts/configure_motor.py --role proximal --yes   # non-interactive
```

The script scans the bus, reports the motor's current state (ID, baudrate, mode, voltage, temperature), and runs the EEPROM unlock → write ID → lock → verify sequence. **Physically label each motor** ("P" or "D") before unplugging — once both are at distinct IDs, it's the only way to tell them apart.

If a motor was previously configured and you don't know its ID, scan the bus:

```bash
uv run python scripts/scan_motors.py                 # full sweep, IDs 1..253
uv run python scripts/scan_motors.py --start 1 --end 10
```

All the gRPC-based scripts below take the gripette endpoint as an explicit argument — there's no default IP. The port defaults to `50051` (gripette's default), so `192.168.1.36` is equivalent to `192.168.1.36:50051`. Replace with the address of your gripette in the examples.

### Teleoperation bridge

Reads angle sensors from the grabette glove (Pi 4) and forwards them as motor commands to the gripper:

```bash
uv run python scripts/teleop_bridge.py --grabette 192.168.1.35 --gripper 192.168.1.36:50051 --dry-run   # preview
uv run python scripts/teleop_bridge.py --grabette 192.168.1.35 --gripper 192.168.1.36:50051            # live
```

`--grabette` accepts `HOST` (defaults to port 8000) or explicit `HOST:PORT`. `--gripper` requires `HOST:PORT`.

### Motor test

Sends a 1Hz sinusoidal command and records feedback positions for delay analysis:

```bash
uv run python scripts/sinus_test.py 192.168.1.36:50051
# Outputs sinus_test.csv (plot inline — see the docstring)
```

For a local equivalent that doesn't go through gRPC, see `scripts/motor_test_local.py`.

### Camera test

Measures stream framerate and saves a sample frame:

```bash
uv run python scripts/camera_test.py 192.168.1.36:50051
# Outputs camera_test.jpg
```

### Reset to zero

Moves both motors to position 0 (fully open):

```bash
uv run python scripts/goto_zero.py 192.168.1.36:50051   # via gRPC
uv run python scripts/goto_zero_local.py                # locally on the Pi, no gRPC
```

## systemd services

`make install-systemd` installs both services (`gripette.service` and `gripette-bluetooth.service`), patching the hard-coded `/home/rasp/Project/Repo/gripette` path in each unit file to this device's actual workspace root.

```bash
make install-systemd          # install + enable both, start now
journalctl -u gripette -f             # main service logs
journalctl -u gripette-bluetooth -f   # BT WiFi-setup service logs
```

### Main service

`gripette.service` runs `python -m gripette` as the `rasp` user — the gRPC motor+camera server.

### Bluetooth WiFi configuration

`gripette-bluetooth.service` is a standalone BLE GATT service that lets you configure WiFi credentials on the enclosed Pi Zero 2W without SSH or a screen. A phone or laptop connects via Bluetooth Low Energy, authenticates with a PIN, and sends WiFi credentials.

Runs as root (required by BlueZ DBus GATT registration). PIN is configurable via `GRIPPER_BT_PIN` env var (default: `00000`). System deps (`python3-dbus`, `python3-gi`) are usually pre-installed on Raspberry Pi OS.

**BLE commands** (written as UTF-8 to the COMMAND characteristic):

| Command | Response | Description |
|---|---|---|
| `PING` | `PONG` | Health check |
| `PIN_xxxxx` | `OK: Connected` / `ERROR: Incorrect PIN` | Authenticate (required before WIFI/WIFI_RESET) |
| `WIFI ssid password` | `OK: Connecting to <ssid>` / `ERROR: ...` | Connect to WiFi via nmcli |
| `WIFI_RESET` | `OK: WiFi connections cleared` | Delete all saved WiFi networks |

Network status is also readable from a dedicated BLE characteristic (auto-updates every 10s).

**Web Bluetooth client**: open the [BT Tool](https://pollen-robotics.github.io/grabette/) in Chrome/Edge on a phone or laptop, then pick **Gripette** in the device chooser (requires HTTPS — it's the single shared page, served via GitHub Pages from `docs/index.html`, and provisions any robot).

See [docs/bluetooth_setup.md](docs/bluetooth_setup.md) for the full setup guide (BlueZ configuration, troubleshooting, etc.).

## Proto definition

The gRPC service contract is defined in `proto/gripper.proto`. To regenerate the Python files after modifying it:

```bash
uv sync --extra dev
uv run python generate_proto.py
```

Generated files in `gripette/proto/` are committed to the repository.
