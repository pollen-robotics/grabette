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

```bash
sudo cp config/config.txt /boot/firmware
```
Then reboot

```bash
sudo apt update && sudo apt install libcap-dev
uv venv --python 3.11 --system-site-packages
uv sync --extra rpi --extra ui
uv run python -m grabette
```

System Python 3.11 is required for `libcamera` access. The `--system-site-packages` flag makes `picamera2` and `numpy` available from the system installation.

### systemd (auto-start on boot)

```bash
sudo cp systemd/grabette.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now grabette

# View logs
journalctl -u grabette -f
```

### Bluetooth WiFi configuration

A standalone BLE GATT service allows configuring WiFi credentials without SSH or a screen. Connect from a phone or laptop via Bluetooth Low Energy, authenticate with a PIN, and send WiFi credentials.

```bash
sudo apt install python3-dbus python3-gi   # system deps (usually pre-installed)
sudo cp systemd/grabette-bluetooth.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now grabette-bluetooth
```

**Important**: set `ControllerMode = le` in `/etc/bluetooth/main.conf` to disable classic Bluetooth (prevents audio profile interference and hostname leak). See the gripette [Bluetooth setup guide](https://github.com/pollen-robotics/gripette/blob/main/docs/bluetooth_setup.md) for detailed instructions.

PIN is configurable via `GRABETTE_BT_PIN` env var (default: `00000`).

**Web Bluetooth client**: open the [BT Tool](https://pollen-robotics.github.io/gripette/) in Chrome/Edge — select "Grabette" from the dropdown to connect (also available locally at `docs/index.html`).

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

GoPro-compatible JSON consumed directly by the UMI SLAM pipeline (ORB-SLAM3):

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
- **IMU brackets video**: IMU starts before first frame, stops before last — required by ORB-SLAM3

## Data Pipeline

```
RPi (camera + BMI088 + AS5600)
  → Grabette service (capture, manage sessions)
  → HuggingFace dataset repo (upload episodes)
  → Cloud SLAM (ORB-SLAM3 via UMI pipeline)
  → Training dataset + 6DoF trajectories
```

## Sibling Projects

| Project | Description |
|---|---|
| [gripette](https://github.com/pollen-robotics/gripette) | gRPC motor+camera service for the motorized gripper (Pi Zero 2W) |
| [grabette-data](https://github.com/pollen-robotics/grabette-data) | grabette data processing (ORB-SLAM3, Docker) |

## Calibration

- **Camera intrinsics**: `../universal_manipulation_interface/example/calibration/rpi_camera_intrinsics.json` (0.41px reproj error)
- **IMU-to-camera transform (T_b_c1)**: 180° rotation around x-axis (back-to-back mounting), 11.15mm translation along z
- **Angle sensor offsets**: `scripts/calibrate_angles.py` → stored in `~/.grabette/angle_calibration.json`
