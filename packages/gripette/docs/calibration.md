# gripette — Calibration (zero offset)

See the [README](../README.md) for install and usage.

A fresh gripette ships with `GRIPPER_MOTOR*_OFFSET=0`, so the encoder's mechanical zero is treated as robot-frame zero. That's usually a few degrees off the gripper's actual "fully open" pose. Calibrate once after assembly to align them.

**From the Bluetooth tool** (recommended; nothing to install, no SSH):

1. Open the [Bluetooth tool](https://pollen-robotics.github.io/grabette/?device=gripette#wifi) in Chrome or Edge, connect to the Gripette in **WiFi setup**, then open the **Calibration** tab.
2. **Release the motors**: the fingers go limp. Hold the gripper if it hangs on a robot arm.
3. Open both fingers **by hand**, all the way, against their stop.
4. **Start calibration** and keep the gripper open: the offsets are saved to `/etc/gripette/env` and the service restarts (a few seconds).
5. **Check the result**: move the fingers by hand, the 3D model should follow. The motors stay released until the robot's program turns the torque back on.

Under the hood, the Bluetooth service (`RELEASE`, `CALIBRATE`, `ANGLES` commands) goes through the gripette service's gRPC API, with the same math as `calibrate_zero.py` below.

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
