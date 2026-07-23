# gripette — Configuration

See the [README](../README.md) for install and usage.

## Robot-frame convention

All motor positions in the API (gRPC, client, scripts, limits) are in **robot frame**:

- `0 rad` — gripper fully **open**
- positive — **closing**
- limits: motor 1 ∈ `[0, +1.484]` (~85°), motor 2 ∈ `[0, +2.025]` (~116°)

Commands outside these bounds are rejected with `ValueError` before reaching the bus. `MotorController` bridges robot frame ↔ encoder frame via per-motor `sign` (from `hand`) and `offset` (from calibration); callers never deal with the encoder values directly.

## Grip force cap & load feedback

`SendMotorCommand` accepts an optional per-motor **torque limit** (`motor{1,2}_torque_limit`, a fraction `0..1` of the servo's max torque). It writes the servo's *running* (RAM) torque-limit register — **not** the EEPROM max-torque register — so it is safe to set every grasp; the value is deduplicated and only written when it changes. A value of `0` means *unset*: the limit is left untouched (full torque), so this is fully backward-compatible — a client that never sets it behaves exactly as before.

Capping the torque turns grip force into a bounded, object-size-independent quantity: the closing joint stalls at the cap instead of pushing to a position error. See the eval-side `--grip_torque_limit` flag in the [simulator/eval README](../../../integrations/openarm/openarm_gripette_simu/README.md).

`MotorState` reports `motor{1,2}_load` — the servo's decoded `present_load` (control effort, ~PWM) in **robot frame: positive = closing effort** (matching the angle convention), consistent across left/right via the per-motor `sign`. The sign is preserved, so an external force driving a joint the *other* way reads negative. Load magnitude clamps at the active torque limit; `present_load` is ~0 whenever torque is disabled. (Sign polarity was calibrated on a right-hand unit — verify on a left unit before trusting its load sign.)

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
