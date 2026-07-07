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

A gripette is built as either a **left** or **right** hand — the motors are mounted mirrored, so the runtime needs to know which one this device is. Pick at install time:

```bash
sudo usermod -aG dialout $USER   # serial bus access — log out + back in for it to take effect
make install-rpi HAND=right       # or HAND=left
sudo reboot                       # required if UART/cmdline were changed
make check                        # post-reboot hardware diagnostic (camera + motors)
```

`HAND` is required — running `make install-rpi` without it fails with a clear error. The choice is written to `/etc/gripette/env` as `GRIPPER_HAND=<value>` and persists across reboots.

`make check` validates the camera and the motor bus. It also probes the two systemd services and reports them as `[SKIP]` if they aren't installed yet — that's the expected state right after `install-rpi`.

Then start the service manually or install at boot:

```bash
uv run --package gripette python -m gripette   # foreground (Ctrl-C to stop)
# — or —
make install-systemd                            # boot-time start (main + bluetooth)
make check                                      # services should now report [OK]
```

`make install-rpi` is idempotent — re-running it is safe (and preserves any captured calibration offsets in `/etc/gripette/env`). Under the hood it:

- installs `python3-libcamera`, `python3-picamera2`, `libcap-dev` via apt;
- runs `make enable-uart` to disable the serial console (`cmdline.txt`) and add `dtoverlay=miniuart-bt` to `config.txt` so the reliable PL011 (`ttyAMA0`) ends up on the GPIO header instead of the mini UART (clock-dependent, unreliable at 1Mbaud);
- runs `make harden-rpi` for crash safety (persistent journal capped at 5 boots / 50MB, and `fsck.mode=force` so a hard shutdown self-heals on next boot — gripettes mounted on a robot get power-cut at random);
- creates a `--system-site-packages` venv at the workspace root so apt's `picamera2` satisfies the dependency tree (otherwise `uv` tries to build `python-prctl` from PyPI);
- runs `uv sync --package gripette --extra rpi --no-install-package numpy` and verifies that `picamera2`, `serial`, and `rustypot` all import;
- writes `GRIPPER_HAND` into `/etc/gripette/env`, preserving any existing `GRIPPER_MOTOR*_OFFSET` lines from a previous calibration.

`make help` lists every target. The cmdline.txt edit captures the `root=PARTUUID=...` token before editing and rolls back from a `.gripette.bak` backup if it changes — boot is safe.

#### Manual installation (fallback)

If `make install-rpi` fails (e.g. unusual OS), the equivalent manual steps are:

1. **UART**: edit `/boot/firmware/config.txt` to include `dtoverlay=miniuart-bt` and `enable_uart=1`. Edit `/boot/firmware/cmdline.txt` to remove `console=serial0,115200` — keep the file as a single line. Reboot.
2. **Deps**: `sudo apt install libcap-dev python3-libcamera python3-picamera2`.
3. **Venv**: from the workspace root, `uv venv --python /usr/bin/python3 --system-site-packages && uv sync --package gripette --extra rpi --no-install-package numpy`.
4. **Hand config**: `sudo mkdir -p /etc/gripette && echo "GRIPPER_HAND=right" | sudo tee /etc/gripette/env` (or `=left`).

## Configuration

### Robot-frame convention

All motor positions in the API (gRPC, client, scripts, limits) are in **robot frame**:

- `0 rad` — gripper fully **open**
- positive — **closing**
- limits: motor 1 ∈ `[0, +1.484]` (~85°), motor 2 ∈ `[0, +2.025]` (~116°)

Commands outside these bounds are rejected with `ValueError` before reaching the bus. `MotorController` bridges robot frame ↔ encoder frame via per-motor `sign` (from `hand`) and `offset` (from calibration); callers never deal with the encoder values directly.

### Environment variables

All settings via environment variables with `GRIPPER_` prefix. Persistent per-device config lives in `/etc/gripette/env`, sourced by `gripette.service`.

| Variable | Default | Description |
|---|---|---|
| `GRIPPER_HOST` | `0.0.0.0` | Server bind address |
| `GRIPPER_PORT` | `50051` | gRPC port |
| `GRIPPER_MOTOR_PORT` | `/dev/serial0` | Serial port for servos |
| `GRIPPER_MOTOR_BAUDRATE` | `1000000` | Serial baudrate |
| `GRIPPER_MOTOR_ID_1` | `1` | First servo ID |
| `GRIPPER_MOTOR_ID_2` | `2` | Second servo ID |
| `GRIPPER_HAND` | `right` | `left` or `right` — determines default `motor*_sign`. Written by `make install-rpi HAND=…` |
| `GRIPPER_MOTOR1_OFFSET` | `0.0` | Encoder reading (rad) at robot-frame zero. Written by `scripts/calibrate_zero_local.py` |
| `GRIPPER_MOTOR2_OFFSET` | `0.0` | Same, motor 2 |
| `GRIPPER_MOTOR1_SIGN` | (from `hand`) | Override the hand-derived sign for motor 1. Use only if a hardware revision is asymmetric. ±1 |
| `GRIPPER_MOTOR2_SIGN` | (from `hand`) | Same, motor 2 |
| `GRIPPER_JPEG_QUALITY` | `70` | JPEG compression quality |
| `GRIPPER_LOG_LEVEL` | `INFO` | Logging level |

## Usage

### Python client

```python
from gripette.client import GripperClient

with GripperClient("192.168.1.36:50051") as g:
    print(g.ping())

    g.torque_on()
    g.move(0.5, 1.0)            # goal positions in radians (0 = open, positive = closing)

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

### Calibration (zero offset)

A fresh gripette ships with `GRIPPER_MOTOR*_OFFSET=0`, so the encoder's mechanical zero is treated as robot-frame zero. That's usually a few degrees off the gripper's actual "fully open" pose. Calibrate once after assembly to align them.

**On the Pi** (recommended for first-time setup; writes `/etc/gripette/env` directly):

```bash
sudo systemctl stop gripette                          # free /dev/serial0
uv run python scripts/calibrate_zero_local.py         # torque off, prompt, write offsets
sudo systemctl start gripette
```

Workflow: torque drops, you physically move the gripper to fully open, press ENTER, the script averages 10 encoder samples and merges `GRIPPER_MOTOR1_OFFSET=…` / `GRIPPER_MOTOR2_OFFSET=…` into `/etc/gripette/env` (preserving `GRIPPER_HAND`). Use `--dry-run` to preview without writing.

**Remote, over gRPC** (no service restart needed; prints values for you to paste):

```bash
uv run python scripts/calibrate_zero.py 192.168.1.36 --hand right
```

Service stays up. The script reads `g.read_motors()` at the user-defined zero pose and prints the **delta** to add to `GRIPPER_MOTOR*_OFFSET` in `/etc/gripette/env`. The delta arithmetic is correct whether this is a first calibration or a re-cal (just add to existing).

### Diagnostics

```bash
uv run python scripts/read_motors.py 192.168.1.36 --torque-off   # live positions, gripper back-drivable
uv run python scripts/scan_motors.py                              # which IDs respond on the bus (run on Pi)
uv run python scripts/configure_motor.py                          # set a brand-new motor's ID (run on Pi)
```

`read_motors.py` is useful for sanity-checking the current calibration: at fully-open the readings should be ~0; at fully-closed they should approach the `motor*_max` limits. See `make check` for the full hardware diagnostic.

---

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
