# Grabette

Part of the GRABETTE project.
Autonomous Raspberry Pi service for robotic manipulation data collection. Captures synchronized camera + IMU streams, manages recording sessions, and integrates with HuggingFace for cloud SLAM processing.

## Hardware

| Component | Spec |
|---|---|
| **Board** | Raspberry Pi 4 |
| **Camera** | RPi camera module, 1296x972 @ 46fps, fisheye lens (KannalaBrandt8) |
| **IMU** | Bosch BMI088, 200Hz, 6-axis (accel + gyro) via I2C |
| **Angle sensors** | 2x AS5600 rotary encoders (proximal + distal joints), I2C buses 4 & 5 |
| **Button** | Grove LED Button (GPIO22 LED, GPIO23 button) — physical start/stop |

Camera and IMU are mounted back-to-back, centers aligned along z-axis, 11.15mm apart.

## Architecture

```
                       ┌──────────────────────────┐
                       │   Web UI (Gradio)         │
                       │   HuggingFace Spaces      │
                       └────────────┬─────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────┐
│                    FastAPI + WebSocket API (:8000)                    │
│                                                                       │
│  /api/state     Live sensor polling + WS stream @10Hz                │
│  /api/camera    JPEG snapshot + WS video stream ~15fps               │
│  /api/episodes  Capture start/stop, download, delete                 │
│  /api/sessions  Session CRUD, episode grouping                       │
│  /api/replay    Episode playback with pause/seek                     │
│  /api/hf        HuggingFace auth, upload, SLAM jobs                  │
│  /api/system    System info, logs, OTA updates                       │
│  /api/daemon    Daemon status + restart                              │
│  /viewer        3D URDF model with live joint angles (Three.js)      │
│  /charts/*      Real-time IMU + angle charts (uPlot)                 │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────┐
│                         Daemon Core                                   │
│          State machine · 50Hz poll loop · Replay engine               │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
                   RpiBackend              MockBackend
                  (real hardware)          (development)
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     VideoCapture  BMI088Capture  AngleCapture
     (picamera2)   (I2C, 200Hz)  (AS5600, I2C)
```

## Quick Start

### Development (mock mode, no hardware needed)

```bash
uv sync
uv run python main.py
# → http://localhost:8000
```

### Raspberry Pi

Tested on **Raspberry Pi OS Bookworm (Debian 12)** and **Trixie (Debian 13)**. No specific Pi OS version is pinned — the Makefile target uses whatever system Python is at `/usr/bin/python3` (3.11 on Bookworm, 3.13 on Trixie).

Prerequisite: install [`uv`](https://docs.astral.sh/uv/), then enable the V2 hardware overlays once (requires reboot):
```bash
sudo cp config/config.txt /boot/firmware
sudo reboot
```

Then one-shot bring-up:
```bash
make install-rpi
uv run python -m grabette
```

`make install-rpi` does the following — automating the steps that are easy to get subtly wrong by hand:
- `sudo apt install python3-libcamera python3-picamera2 libcap-dev ffmpeg`
- Installs the OAK-D / Movidius USB udev rule (`/etc/udev/rules.d/80-movidius.rules`)
- Creates the venv with `uv venv --python /usr/bin/python3 --system-site-packages` — **both flags matter**:
  - `--python /usr/bin/python3` ensures uv uses the apt-managed Python (which owns `python3-libcamera`/`python3-picamera2`), not uv's own managed Python under `~/.local/share/uv/python/...`.
  - `--system-site-packages` makes the apt-installed `libcamera` and `numpy` visible to the venv.
- Runs `uv sync --extra rpi --extra ui` and verifies all imports succeed.

If the daemon logs `Using MockBackend` instead of `RPi hardware detected, using RpiBackend`, the venv setup didn't take — `make install-rpi` will fix it on a re-run.

### systemd (auto-start on boot)

```bash
make install-systemd
journalctl -u grabette -f   # logs
```

### Bluetooth WiFi configuration

A standalone BLE GATT service allows configuring WiFi credentials without SSH or a screen. To make it operational : 

```bash
make install-rpi
```

Then, connect from a phone or laptop via Bluetooth Low Energy on the [BT Tool](https://pollen-robotics.github.io/grabette/) in Chrome/Edge and follow those steps : 
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


No pairing is required: the characteristics are unencrypted and the adapter advertises with `Pairable = False`, so clients connect "connection-only" — no pairing dialog, no terminal needed.

> **If a client still tries to pair** (e.g. a stale bond from an earlier version): clear it on both ends — `bluetoothctl remove <mac.address.of.Grabette>` on the Pi and the client, plus Forget the device in `chrome://bluetooth-internals`. As a last resort, run an auto-accept agent on the client: in a terminal, `bluetoothctl` → `agent NoInputNoOutput` → `default-agent` (leave it open).

## Configuration

All settings via environment variables with `GRABETTE_` prefix:

| Variable | Default | Description |
|---|---|---|
| `GRABETTE_HOST` | `0.0.0.0` | Server bind address |
| `GRABETTE_PORT` | `8000` | Server port |
| `GRABETTE_BACKEND` | `auto` | `auto`, `mock`, or `rpi` |
| `GRABETTE_DATA_DIR` | `~/grabette-data` | Data storage directory |
| `GRABETTE_CAMERA_FPS` | `46` | Camera frame rate |
| `GRABETTE_IMU_HZ` | `200` | IMU sample rate |
| `GRABETTE_ANGLE_SENSORS` | `true` | Enable AS5600 angle sensors |
| `GRABETTE_UI_ENABLED` | `true` | Enable Gradio dashboard |
| `GRABETTE_BUTTON_ENABLED` | `true` | Enable hardware button |
| `GRABETTE_LOG_LEVEL` | `INFO` | Logging level |

## Data

### Organization

Two-level hierarchy: **sessions** (named groups) containing **episodes** (individual captures).

```
~/grabette-data/
├── sessions.json                    # Session registry
└── episodes/
    └── 20260310_143052/             # One episode
        ├── raw_video.mp4            # H.264 encoded (1296x972 @ 46fps)
        ├── imu_data.json            # BMI088 accel + gyro (200Hz)
        ├── angle_data.json          # AS5600 joint angles (if available)
        └── metadata.json            # Duration, frame count, sample counts
```

### IMU format

GoPro-compatible JSON (ACCL/GYRO streams) consumed by the SLAM/VIO pipeline:

```json
{
  "1": {
    "streams": {
      "ACCL": { "samples": [{"cts": 0.0, "value": [x, y, z]}] },
      "GYRO": { "samples": [{"cts": 0.0, "value": [x, y, z]}] }
    }
  }
}
```

Units: accel in m/s² (includes gravity), gyro in rad/s. Timestamps in milliseconds from video start.

### Capture synchronization

All sensor streams share a common `SyncManager` clock based on `time.monotonic()`:

- **Camera**: SensorTimestamp from picamera2 (same SoC hardware clock — no drift)
- **IMU**: BMI088 SENSORTIME register (internal oscillator, ~1% drift) — corrected via two-point linear rescaling at capture stop
- **Contention prevention**: `_capturing` flag blocks daemon I2C reads during recording
- **Stop order**: IMU first, then camera (camera stop includes ffmpeg muxing)
- **IMU brackets video**: IMU starts before first frame, stops before last — required by the downstream SLAM/VIO pipeline

## Data Pipeline

```
RPi (camera + BMI088 + AS5600)
  → Grabette service (capture, manage sessions)
  → HuggingFace dataset repo (upload episodes)
  → Cloud SLAM/VIO processing
  → Training dataset + 6DoF trajectories
```

## Sibling Projects

| Project | Description |
|---|---|
| [gripette](https://github.com/pollen-robotics/gripette) | gRPC motor+camera service for the motorized gripper (Pi Zero 2W) |
| [grabette-data](https://github.com/pollen-robotics/grabette-data) | grabette data processing (SLAM/VIO + LeRobot dataset generation, Docker) |

## Calibration

- **Camera intrinsics**: `../universal_manipulation_interface/example/calibration/rpi_camera_intrinsics.json` (0.41px reproj error)
- **IMU-to-camera transform (T_b_c1)**: 180° rotation around x-axis (back-to-back mounting), 11.15mm translation along z
- **Angle sensor offsets**: `scripts/calibrate_angles.py` → stored in `~/.grabette/angle_calibration.json`
