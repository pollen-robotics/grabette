# gripette — Scripts & diagnostics

Utility scripts for a running gripette. See the [README](../README.md) for install and usage, and [motor_setup.md](motor_setup.md) / [calibration.md](calibration.md) for first-time bring-up.

## Diagnostics

```bash
uv run python scripts/read_motors.py 192.168.1.36 --torque-off   # live positions, gripper back-drivable
uv run python scripts/scan_motors.py                              # which IDs respond on the bus (run on Pi)
uv run python scripts/configure_motor.py                          # set a brand-new motor's ID (run on Pi)
```

`read_motors.py` is useful for sanity-checking the current calibration: at fully-open the readings should be ~0; at fully-closed they should approach the `motor*_max` limits. See `make check` for the full hardware diagnostic.

---

All the gRPC-based scripts below take the gripette endpoint as an explicit argument — there's no default IP. The port defaults to `50051` (gripette's default), so `192.168.1.36` is equivalent to `192.168.1.36:50051`. Replace with the address of your gripette in the examples.

## Teleoperation bridge

Reads angle sensors from the grabette glove (Pi 4) and forwards them as motor commands to the gripper:

```bash
uv run python scripts/teleop_bridge.py --grabette 192.168.1.35 --gripper 192.168.1.36:50051 --dry-run   # preview
uv run python scripts/teleop_bridge.py --grabette 192.168.1.35 --gripper 192.168.1.36:50051            # live
```

`--grabette` accepts `HOST` (defaults to port 8000) or explicit `HOST:PORT`. `--gripper` requires `HOST:PORT`.

## Motor test

Sends a 1Hz sinusoidal command and records feedback positions for delay analysis:

```bash
uv run python scripts/sinus_test.py 192.168.1.36:50051
# Outputs sinus_test.csv (plot inline — see the docstring)
```

For a local equivalent that doesn't go through gRPC, see `scripts/motor_test_local.py`.

## Camera test

Measures stream framerate and saves a sample frame:

```bash
uv run python scripts/camera_test.py 192.168.1.36:50051
# Outputs camera_test.jpg
```

## Reset to zero

Moves both motors to position 0 (fully open):

```bash
uv run python scripts/goto_zero.py 192.168.1.36:50051   # via gRPC
uv run python scripts/goto_zero_local.py                # locally on the Pi, no gRPC
```
