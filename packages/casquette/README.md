# Casquette 🚧 (Work in Progress)

> **Casquette is an experimental subproject, still under active development —
> not ready to build or deploy.** The hardware and software are still changing,
> so we don't recommend building one yet. For data collection today, use
> [grabette](../grabette).

POV (point-of-view) head-mounted capture: the planned egocentric counterpart to
the handheld [grabette](../grabette) — the same capture idea, worn on the head to
record a first-person view during manipulation tasks.

## Planned hardware

- Raspberry Pi Zero 2W
- RPi camera module (1296x972 @ 46fps, fisheye lens)
- BMI088 IMU (200Hz, accel + gyro)

## Development (mock mode, no hardware)

You can run the service against a mock backend to explore the code:

```bash
uv sync --package casquette
uv run --package casquette python main.py   # → http://localhost:8001
```

On-device build, systemd, and API docs are intentionally omitted while the design
is in flux. For a working device and the full capture → postprocess pipeline, see
[grabette](../grabette) and [grabette-postprocess](../grabette-postprocess).
