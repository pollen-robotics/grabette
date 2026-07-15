# gripette — Configuration

See the [README](../README.md) for install and usage.

## Robot-frame convention

All motor positions in the API (gRPC, client, scripts, limits) are in **robot frame**:

- `0 rad` — gripper fully **open**
- positive — **closing**
- limits: motor 1 ∈ `[0, +1.484]` (~85°), motor 2 ∈ `[0, +2.025]` (~116°)

Commands outside these bounds are rejected with `ValueError` before reaching the bus. `MotorController` bridges robot frame ↔ encoder frame via per-motor `sign` (from `hand`) and `offset` (from calibration); callers never deal with the encoder values directly.

## Environment variables

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
