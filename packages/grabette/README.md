# Grabette

Autonomous Raspberry Pi service for robotic manipulation data collection. It captures synchronized camera + depth + IMU streams from a handheld gripper, manages recording sessions, and uploads episodes to Hugging Face for cloud SLAM processing. Part of the [GRABETTE project](../../README.md).

## Hardware

| Component | Spec |
|---|---|
| **Board** | Raspberry Pi 4 |
| **Primary camera** | RPi camera module, 1296x972 @ 46fps, fisheye lens (KannalaBrandt8) |
| **OAK-D SR** | Stereo RGB-D camera with on-board BNO IMU (200Hz). Provides the depth + IMU stream for SLAM — **required** for trajectory recovery on Grabette. Replaces the legacy BMI088. Toggled on demand (default off to save battery; turn it on when recording for the pipeline). |
| **Angle sensors** | 2x AS5600L rotary encoders (proximal + distal finger joints), one per I2C bus (`/dev/i2c-3` distal, `/dev/i2c-4` proximal) |
| **Button** | Grove LED Button (GPIO22 LED, GPIO23 button) — physical start/stop |

📋 **[Full Bill of Materials (BOM)](https://docs.google.com/spreadsheets/d/e/2PACX-1vQ3LyyWI-CiplVPtgrWkmLRYjdDqYhbVJXYt8PNa71FDzbTSMVj1YGV0Zpo5PJeBGJURaz8nZt1_v-8/pubhtml)** — complete parts list (shared for Grabette + Gripette).
🧩 **[CAD — Onshape](https://cad.onshape.com/documents/0c6175c392788391992ff2ec/w/9f773e5f0eeae1577ae36a05/e/13a89fef2591d863bb0bf186)** — full Grabette + Gripette CAD.

## Install

### Development (mock mode, no hardware needed)

> Part of the uv **workspace**: a bare `uv sync` here would build the *entire
> monorepo* environment. Always pass `--package` (root README → Development).

```bash
uv sync --package grabette
uv run --package grabette python main.py
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

To put the device on WiFi without a screen or SSH, use the BLE setup service — see **[docs/bluetooth_setup.md](docs/bluetooth_setup.md)**.

## Usage

Once running (mock or on-device), open the dashboard at `http://<device>:8000`. From there — or with the hardware button — you can:

1. Preview the camera and live sensor charts.
2. Start/stop a recording (press the button, or use the UI / `/api/episodes`). Episodes are grouped into named sessions.
3. Review or replay captured episodes.
4. Upload episodes to a Hugging Face dataset repo and trigger cloud SLAM (`/api/hf`).

Recordings are written to `~/grabette-data/` — see the [data format](docs/data_format.md). Downstream SLAM → LeRobot dataset generation is handled by [grabette-postprocess](../grabette-postprocess).

## Documentation

- [Architecture](docs/architecture.md) — daemon internals, API surface, backends.
- [Configuration](docs/configuration.md) — environment variables and the robot-frame angle convention.
- [Data format](docs/data_format.md) — episode layout, calibration & geometry, IMU format, synchronization.
- [Bluetooth WiFi setup](docs/bluetooth_setup.md) — headless WiFi provisioning over BLE.

## Related packages

| Package | Description |
|---|---|
| [gripette](../gripette) | gRPC motor+camera service for the motorized gripper (Pi Zero 2W) |
| [grabette-postprocess](../grabette-postprocess) | SLAM/VIO + LeRobot dataset generation (Docker) |
