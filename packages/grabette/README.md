# Grabette

Part of the GRABETTE project.
Autonomous Raspberry Pi service for robotic manipulation data collection. Captures synchronized camera + IMU streams, manages recording sessions, and integrates with HuggingFace for cloud SLAM processing.

## Hardware

| Component | Spec |
|---|---|
| **Board** | Raspberry Pi 4 |
| **Primary camera** | RPi camera module, 1296x972 @ 46fps, fisheye lens (KannalaBrandt8) |
| **OAK-D SR** | Stereo RGB-D camera with on-board BNO IMU (200Hz). Provides depth + IMU stream for SLAM; replaces the legacy BMI088. Toggled on demand (default off to save battery). |
| **Angle sensors** | 2x AS5600L rotary encoders (proximal + distal finger joints), one per I2C bus (`/dev/i2c-3` distal, `/dev/i2c-4` proximal) |
| **Button** | Grove LED Button (GPIO22 LED, GPIO23 button) — physical start/stop |

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
     VideoCapture   OakdCapture   AngleCapture
     (picamera2)    (RGB-D + IMU,  (AS5600L, I2C)
                     toggleable)
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

Prerequisite: install [`uv`](https://docs.astral.sh/uv/), then enable the V2 hardware overlays once and grant rights for network scanning (requires reboot):
```bash
sudo cp config/config.txt /boot/firmware
make install-netdev
sudo reboot
```

Then one-shot bring-up. A grabette is built as either a **left** or **right** hand — the angle sensors are mounted mirrored, so the daemon needs to know which one this device is. Pick at install time:
```bash
make install-rpi HAND=right    # or HAND=left
uv run python -m grabette
```

`HAND` is required — running `make install-rpi` without it fails with a clear error. The choice is written to `/etc/grabette/env` as `GRABETTE_HAND=<value>` and persists across reboots (sourced by `grabette.service`).

`make install-rpi HAND=...` does the following — automating the steps that are easy to get subtly wrong by hand:
- `sudo apt install python3-libcamera python3-picamera2 libcap-dev ffmpeg python3-dbus python3-gi` (the dbus/gi packages are system deps for the BLE WiFi service).
- Installs the OAK-D / Movidius USB udev rule (`/etc/udev/rules.d/80-movidius.rules`).
- Creates the venv with `uv venv --python /usr/bin/python3 --system-site-packages` — **both flags matter**:
  - `--python /usr/bin/python3` ensures uv uses the apt-managed Python (which owns `python3-libcamera`/`python3-picamera2`), not uv's own managed Python under `~/.local/share/uv/python/...`.
  - `--system-site-packages` makes the apt-installed `libcamera` and `numpy` visible to the venv.
- Runs `uv sync --package grabette --extra rpi --extra ui --extra hf` and verifies all imports succeed.
- Writes `/etc/grabette/env` with `GRABETTE_HAND=<value>` (preserving any prior `GRABETTE_*_SIGN` overrides).

Note: `install-rpi` does **not** install or start the systemd services — that's `make install-systemd` (next section).

If the daemon logs `Using MockBackend` instead of `RPi hardware detected, using RpiBackend`, the venv setup didn't take — `make install-rpi` will fix it on a re-run.

### systemd (auto-start on boot)

`make install-systemd` installs **both** services (`grabette.service` and `grabette-bluetooth.service`), runs `ensure-ble-only` to set BlueZ to `ControllerMode = le`, then `enable --now`s them so they're up immediately and across reboots.

```bash
make install-systemd
journalctl -u grabette -f               # daemon logs
journalctl -u grabette-bluetooth -f     # BLE WiFi-setup service logs
```

If you re-run `install-systemd` while the services are already up, `enable --now` does NOT restart them — issue `sudo systemctl restart grabette grabette-bluetooth` to pick up updated unit files.

### Bluetooth WiFi configuration

A standalone BLE GATT service (`grabette-bluetooth.service`) lets you configure WiFi credentials without SSH or a screen. It's installed + started by `make install-systemd` (above). Once running:

Connect from a phone or laptop via Bluetooth Low Energy on the [BT Tool](https://pollen-robotics.github.io/grabette/) in Chrome/Edge and follow those steps : 
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

## Configuration

### Robot-frame convention

Finger angles published in `AngleSample.proximal` / `AngleSample.distal` (and in the data this daemon writes) are in **robot frame**, matching the gripette runtime:

- `0 rad` — fingers fully **open**
- positive — **closing**

The two AS5600L magnets rotate in opposite directions when the fingers close, and a right-hand grabette is the mirror of a left-hand one — so the per-sensor sign that bridges raw rotation → robot frame depends on the `hand` setting. Defaults: `right → distal=+1, proximal=-1`; `left → distal=-1, proximal=+1`. Override individual signs via `GRABETTE_DISTAL_SIGN` / `GRABETTE_PROXIMAL_SIGN` only for an asymmetric hardware revision.

### Environment variables

All settings via environment variables with `GRABETTE_` prefix. Persistent per-device config lives in `/etc/grabette/env`, sourced by `grabette.service`.

| Variable | Default | Description |
|---|---|---|
| `GRABETTE_HOST` | `0.0.0.0` | Server bind address |
| `GRABETTE_PORT` | `8000` | Server port |
| `GRABETTE_BACKEND` | `auto` | `auto`, `mock`, or `rpi` |
| `GRABETTE_DATA_DIR` | `~/grabette-data` | Data storage directory |
| `GRABETTE_CAMERA_FPS` | `46` | Camera frame rate |
| `GRABETTE_IMU_HZ` | `200` | IMU sample rate |
| `GRABETTE_ANGLE_SENSORS` | `true` | Enable AS5600 angle sensors |
| `GRABETTE_HAND` | `right` | `left` or `right` — determines default `*_sign`. Written by `make install-rpi HAND=…` |
| `GRABETTE_DISTAL_SIGN` | (from `hand`) | Override the hand-derived distal sensor sign. ±1 |
| `GRABETTE_PROXIMAL_SIGN` | (from `hand`) | Override the hand-derived proximal sensor sign. ±1 |
| `GRABETTE_UI_ENABLED` | `true` | Enable Gradio dashboard |
| `GRABETTE_BUTTON_ENABLED` | `true` | Enable hardware button |
| `GRABETTE_LOG_LEVEL` | `INFO` | Logging level |

## Data

### Organization

Two-level hierarchy: **sessions** (named groups) containing **episodes** (individual captures).

```
~/grabette-data/
├── sessions.json                       # Session registry
└── episodes/
    └── 20260310_143052/                # One episode
        ├── raw_video.mp4               # Primary RPi cam, H.264 (1296x972 @ 46fps)
        ├── frame_timestamps.json       # Per-frame timestamps for raw_video
        ├── imu_data.json               # OAK-D IMU: accel + gyro + rotation (200Hz)
        ├── angle_data.json             # AS5600L joint angles (~85–100Hz)
        ├── rpi_camera_intrinsics.json  # Fisheye KB8 calibration for the primary cam
        ├── frames.json                 # URDF-derived frame transforms, incl. T_camera_in_oak_l
        ├── oakd_left.mp4               # OAK-D stereo left (H.264)
        ├── oakd_right.mp4              # OAK-D stereo right (H.264)
        ├── oakd_depth.mkv              # OAK-D depth stream
        ├── oakd_*_timestamps.json      # Per-stream timestamps
        ├── oakd_calib.json             # OAK-D factory EEPROM dump
        ├── oakd_calib_offline.json     # Flat fx/fy/cx/cy/baseline/imu_to_cam for SLAM
        ├── oakd_clock_pairs.json       # OAK-D ↔ SyncManager clock alignment
        └── metadata.json               # Duration, counts, hand, angle_convention, device_id, urdf
```

**Per-episode calibration + geometry** (added by the rpi backend at capture stop):

- `rpi_camera_intrinsics.json` — copied from `config/rpi_camera_intrinsics.json` (KannalaBrandt8 fisheye model, ~0.32px reproj). Ships as a single canonical file for all devices; per-device calibration is a separate open task.
- `frames.json` — computed from `urdf/grabette_{hand}/robot.urdf` at capture stop. Contains each frame's 4×4 transform in the `grip_r` link frame (`camera`, `oak_l`, `oak_r`, `gripper_center`, `thumb_tip`) plus the pre-composed `T_camera_in_oak_l` (so SLAM poses produced in the `oak_l` frame can be re-expressed in the primary camera frame without URDF parsing on the consumer side).
- `metadata.json.urdf` — records which URDF was used (`grabette_left` / `grabette_right`) for traceability.

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
- **IMU**: depthai timestamps from the OAK-D pipeline, mapped onto the SyncManager clock at sample arrival
- **Contention prevention**: `_capturing` flag blocks daemon I2C reads during recording
- **Stop order**: IMU/depth first, then camera (camera stop includes ffmpeg muxing)
- **IMU brackets video**: IMU starts before first frame, stops before last — required by the downstream SLAM/VIO pipeline

## Data Pipeline

```
RPi (camera + OAK-D + AS5600L)
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
