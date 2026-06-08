# Casquette

POV (point-of-view) camera data collection service for the [GRABETTE](https://github.com/SteveNguyen/grabette) project. Captures synchronized camera + IMU data from the user's head-mounted perspective during manipulation tasks.

Runs on a Raspberry Pi Zero 2W with a camera module and BMI088 IMU.

## Hardware

- Raspberry Pi Zero 2W
- RPi camera module (1296x972 @ 46fps, fisheye lens)
- BMI088 IMU (200Hz, accel + gyro) on I2C bus 1

## Quick Start

### Development (mock mode, no hardware needed)

```bash
uv sync
uv run python main.py
# → http://localhost:8001
```

### Raspberry Pi Zero 2W

```bash
sudo cp config/config.txt /boot/firmware/
# Edit /boot/firmware/cmdline.txt: remove "console=serial0,115200"
# Reboot

uv venv --python /usr/bin/python3 --system-site-packages
uv sync --extra rpi --no-install-package numpy
uv pip install -e .
uv run python -m casquette
```

### systemd (auto-start on boot)

```bash
sudo cp systemd/casquette.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now casquette

journalctl -u casquette -f
```

### Bluetooth WiFi configuration

```bash
sudo apt install python3-dbus python3-gi
sudo cp systemd/casquette-bluetooth.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now casquette-bluetooth
```

Set `ControllerMode = le` in `/etc/bluetooth/main.conf`. See the gripette [Bluetooth setup guide](https://github.com/SteveNguyen/gripette/blob/main/docs/bluetooth_setup.md).

## Configuration

All settings via environment variables with `CASQUETTE_` prefix (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `CASQUETTE_HOST` | `0.0.0.0` | Server bind address |
| `CASQUETTE_PORT` | `8001` | Server port |
| `CASQUETTE_BACKEND` | `auto` | `auto`, `mock`, or `rpi` |
| `CASQUETTE_DATA_DIR` | `~/casquette-data` | Data storage directory |
| `CASQUETTE_CAMERA_FPS` | `46` | Camera frame rate |
| `CASQUETTE_IMU_HZ` | `200` | IMU sample rate |
| `CASQUETTE_IMU_I2C_BUS` | `1` | I2C bus for BMI088 |
| `CASQUETTE_DEVICE_ID` | `""` | Device ID (for multi-device sync) |
| `CASQUETTE_LOG_LEVEL` | `INFO` | Logging level |

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/daemon/status` | GET | Daemon state |
| `/api/state` | GET | Current sensor state |
| `/api/state/ws` | WS | Live sensor stream (10Hz) |
| `/api/camera/snapshot` | GET | JPEG snapshot |
| `/api/camera/ws` | WS | Video stream (~15fps) |
| `/api/episodes/start` | POST | Start recording |
| `/api/episodes/stop` | POST | Stop recording |
| `/api/episodes/{id}` | GET | Episode info |
| `/api/episodes/{id}/download` | GET | Download episode (tar.gz) |
| `/api/episodes/{id}/video` | GET | Stream video |
| `/api/sessions` | GET | List sessions |
| `/api/system/info` | GET | System information |

## Data Format

Episodes are stored in `~/casquette-data/episodes/{timestamp}/`:

```
20260326_143022/
├── raw_video.mp4      # H.264 (1296x972 @ ~46fps)
├── imu_data.json      # GoPro-compatible (ACCL + GYRO streams, 200Hz)
└── metadata.json      # Duration, counts, device_id, wall_clock_start_utc
```

Compatible with the [grabette-data](https://github.com/SteveNguyen/grabette-data) SLAM pipeline.

## Sibling Projects

| Project | Description |
|---|---|
| [grabette](https://github.com/SteveNguyen/grabette) | Main data collection service (camera + IMU + angle sensors) |
| [gripette](https://github.com/SteveNguyen/gripette) | Motorized gripper service (camera + servos) |
| [grabette-data](https://github.com/SteveNguyen/grabette-data) | SLAM processing + LeRobot dataset generation |
