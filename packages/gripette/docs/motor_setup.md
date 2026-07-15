# gripette — Motor setup & assembly

See the [README](../README.md) for install and usage.

A gripette uses two Feetech STS3215 servos with distinct IDs. Brand-new motors all ship as ID=1 at 1Mbaud in position mode, so for each new gripper one of the two motors must be reconfigured before assembly.

| role     | motor_id | physical position |
|----------|----------|-------------------|
| proximal | 1        | base of the finger |
| distal   | 2        | tip of the finger  |

Use `configure_motor.py` to set each motor's ID. Connect **one motor at a time** on the bus (two motors both at ID=1 collide and the bus returns nothing usable):

```bash
uv run python scripts/configure_motor.py             # interactive: prompts for role
uv run python scripts/configure_motor.py --info      # read-only: prints current config
uv run python scripts/configure_motor.py --role proximal --yes   # non-interactive
```

The script scans the bus, reports the motor's current state (ID, baudrate, mode, voltage, temperature), and runs the EEPROM unlock → write ID → lock → verify sequence. **Physically label each motor** ("P" or "D") before unplugging — once both are at distinct IDs, it's the only way to tell them apart.

If a motor was previously configured and you don't know its ID, scan the bus:

```bash
uv run python scripts/scan_motors.py                 # full sweep, IDs 1..253
uv run python scripts/scan_motors.py --start 1 --end 10
```
