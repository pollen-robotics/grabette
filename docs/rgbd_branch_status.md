# Grabette V2 (rgbd branch) — status & resume guide

Working notes for the V2 hardware migration. Pick this up to continue without losing context.

## Goal

Migrate from V1 (Grove HAT + bit-banged I2C + BMI088 IMU) to V2 (custom HAT + hardware I2C + OAK-D SR over USB3). The OAK-D SR replaces the BMI088 and adds RGBD streams. The RPi fisheye camera, AS5600 angle sensors, and a switch/LED stay.

## Hardware summary (V2)

| Function | Bus / interface | Pins | Address | Notes |
|---|---|---|---|---|
| RPi camera (fisheye) | CSI | — | — | OV5647, 1296x972 @ 50fps, unchanged from V1 |
| AS5600 distal | hw I2C3 | GPIO 4 SDA / 5 SCL | 0x36 | `dtoverlay=i2c3,pins_4_5` |
| AS5600 proximal | hw I2C4 | GPIO 8 SDA / 9 SCL | 0x36 | `dtoverlay=i2c4,pins_8_9` |
| Switch | direct GPIO | GPIO 10 | — | input, active-low, internal pull-up |
| LED | direct GPIO | GPIO 11 | — | output, active-high |
| OAK-D SR | USB3 | — | — | stereo + depth + IMU |
| BMI088 (on HAT, unused) | hw I2C1 | GPIO 2/3 | 0x18, 0x19, 0x68 | leftover on HAT, ignore |
| Audio codec (on HAT, unused) | hw I2C1 | GPIO 2/3 | 0x40 | not used in software |

Power: device currently runs from a USB-C wall supply. PiSugar 3+ for V1 prototypes. A custom 2S battery pack with dual buck regulators is planned (Pi rail + OAK-D rail, common ground).

## Phase 1 — DONE

Mechanical changes: hardware I2C overlays, drop BMI088, switch/LED on direct GPIO. All committed to `rgbd` branch (commit `f4872ab` "wip" or later).

| Change | Files |
|---|---|
| Hardware I2C overlays for AS5600s | `config/config.txt` |
| Default I2C bus numbers 4/5 → 3/4 | `grabette/hardware/angle.py` |
| LedButton default pins 22/23 → 11/10 | `grabette/hardware/button.py` |
| Drop BMI088 path | `grabette/backend/rpi.py` (rewrite), deleted `grabette/hardware/imu.py` and `grabette/hardware/bmi088.py` |
| Save `frame_timestamps.json` and `angle_data.json` per episode (no more `imu_data.json` from rpi backend) | `grabette/backend/rpi.py` |

### Deployed and verified on device (rgrabette2 @ 192.168.1.19)

- New config.txt deployed, device rebooted: `/dev/i2c-3` and `/dev/i2c-4` come up correctly with `i2c_bcm2835` driver.
- Both buses functional (audio codec at `0x40` and BMI088 at `0x18/0x19/0x68` on i2c-1 detected).
- OAK-D recognized on USB but at **480 Mbps (USB 2.0)** — see open issues.

### Known downstream breakage (acceptable on this branch, fix in Phase 2)

- `grabette/replay.py`, `grabette/app/routers/replay.py` still hard-require `imu_data.json` — replay tool will fail on V2 episodes until updated.
- `grabette/hf.py`, `grabette/session.py` reference `imu_data.json` in comments/docstrings; runtime falls back gracefully.
- `grabette/backend/mock.py` still writes fake `imu_data.json` — fine, it's a dev tool.

## Open blocker — AS5600 sensors not detected

`i2cdetect -y 3` and `i2cdetect -y 4` do **not** show 0x36 on either bus. The hardware I2C peripherals themselves work (bus alive, audio codec at 0x40 acknowledges). Three things to investigate before any more software work:

1. **Power**: measure 3.3V on the AS5600 VCC pins with a multimeter (with everything plugged in and powered).
2. **Pull-ups**: BCM2711 hardware I2C peripherals have **no internal pull-ups**. The HAT must include external pull-ups (4.7kΩ or 10kΩ to 3.3V on each SDA and SCL line). Measure SDA-to-3.3V and SCL-to-3.3V — should be near 3.3V when bus is idle.
3. **Wiring continuity**: trace from RPi GPIO 4/5/8/9 to the corresponding SDA/SCL pins on each AS5600. Common mistake: SDA/SCL swapped (chip won't ACK).

The fact that `0x40` shows on *both* i2c-3 and i2c-4 is suspicious. It may be a phantom response on a floating bus (no/weak pull-ups), or the HAT actually wires the codec to multiple buses (would be unusual).

## Open issue — OAK-D running at USB 2.0

`lsusb -t` shows OAK-D (`03e7:2485 Intel Movidius MyriadX`) on Bus 001 at 480 Mbps. Bus 002 (USB 3.0, 5000 Mbps) is empty. For stereo + depth at 30fps we need USB 3.0.

Likely causes (in priority):
1. OAK-D plugged into a black USB 2.0 port instead of a blue USB 3.0 port on the Pi.
2. USB-C cable to OAK-D is USB 2.0 only (most generic ones are).
3. Insufficient bus power (Pi is currently powered via USB-C, may sag under combined load — `vcgencmd get_throttled` is `0x0` for now but only catches sustained undervoltage).

## Phase 2 — TODO

When Phase 1 hardware is fully verified (AS5600 reads + OAK-D on USB3), proceed with:

### `grabette` repo (rgbd branch)

- **`grabette/hardware/oakd.py`** — new module wrapping depthai. Pipeline: 2× mono cams + StereoDepth + IMU. Capture in a thread, queue frames to writers. Writers:
  - left/right mono → H.264 mp4 via on-device hardware encoder (low CPU)
  - depth → uint16 PNG sequence
  - IMU → JSON in GoPro-format (drop-in for our existing tooling)
  - Pair device-side and host-side timestamps periodically for clock translation
  - Save factory calibration JSON once at start
- **Sync update**: keep `SyncManager` (monotonic) as master; OAK-D timestamps stored as `{device_ts, host_ts}` pairs; offline tools fit linear (drift) correction same as we did for BMI088.
- **`grabette/backend/rpi.py`** — wire `oakd.py` into the capture session. Update `start_capture` / `stop_capture` to drive it alongside the RPi camera and angle sensors.
- **Output schema** per episode after Phase 2:
  ```
  raw_video.mp4              RPi fisheye (unchanged)
  metadata.json              capture summary
  frame_timestamps.json      RPi camera frame ts (already in Phase 1)
  angle_data.json            proximal+distal (already in Phase 1)
  oakd_left.mp4              1280x800 mono left, H.264 hw-encoded
  oakd_right.mp4             1280x800 mono right, H.264 hw-encoded
  oakd_depth/                uint16 PNG sequence, 1280x800
  oakd_depth_timestamps.json one ts per depth frame (host_ts)
  oakd_imu.json              ACCL+GYRO, GoPro-format
  oakd_calib.json            factory intrinsics+extrinsics
  oakd_clock_pairs.json      device_ts ↔ host_ts pairs
  ```
- **Restore replay/upload paths**: update `replay.py`, `app/routers/replay.py`, `hf.py` to handle the new schema (no more `imu_data.json` from rpi backend; OAK-D files in their place).

### `grabette-data` repo (rgbd branch — already created by user)

- **MCAP converter**: episode dir → MCAP for RTAB-Map ingestion.
- **Offline RTAB-Map pipeline**: docker-based, batch script analogous to current `batch_slam.py`. RTAB-Map chosen as primary SLAM (offline). Inputs: stereo + depth + IMU.
- **Optional ORB-SLAM3 stereo-inertial path**: keep as alternative; same recordings, different SLAM.
- **`generate_dataset.py` extension**: add OAK-D streams as additional `observation.images.*` features (RGB and/or depth).

## Decisions made (reference)

- **SLAM**: offline only (no live SLAM on RPi 4 — avoid CPU contention during capture). RTAB-Map is the primary choice; ORB-SLAM3 stereo-inertial as backup option.
- **Recording**: stereo + depth + IMU + RPi fisheye + angle sensors, all raw streams, no SLAM during capture.
- **Resolution**: OAK-D stereo at 1280x800 native (sweet spot for the SR's calibrated stereo matcher). SLAM can downsample to 640x400 offline if needed.
- **Depth**: record uint16 PNG sequence (~600 MB/min). Drop later if disk pressure forces it (we can recompute offline from stereo).
- **Synchronization**: `SyncManager` (monotonic) is master. OAK-D timestamps translated via paired ts samples. Multi-Grabette is future work (chrony as easy default; GPS PPS as upgrade; CM4 with hw PTP if sub-µs ever needed).
- **Power on custom HAT**: plan dual 5V rails from 2S battery (one for Pi, one for OAK-D), common ground; ≥4A buck per rail; INA226 per rail for current monitoring; LiFePO4 for safety vs LiPo for energy density.
- **`enable_uart=1`**: not in repo config; only added on the deployed device when needed.

## Resume checklist

When picking this up:

1. **Hardware first** — until AS5600 detection works, software changes are blind:
   - [ ] Verify 3.3V at AS5600 VCC pins (multimeter)
   - [ ] Verify pull-ups present on each I2C bus (~3.3V at idle)
   - [ ] Verify SDA/SCL not swapped from RPi GPIO to AS5600 pins
   - [ ] Once corrected: `i2cdetect -y 3` and `i2cdetect -y 4` should each show `0x36`
2. **OAK-D on USB 3.0** — move to a blue port with a known USB 3 cable. `lsusb -t` should show device on Bus 002 at 5000M.
3. **Power** — decide between USB-C wall supply, dedicated PSU, or starting on the 2S design. Current setup is marginal under full load.
4. **Then Phase 2** — start with `grabette/hardware/oakd.py` (see TODO list above).

## Quick commands

```bash
# Local dev machine — repo
cd /home/steve/Project/Repo/GRABETTE/grabette
git checkout rgbd
git status

# Device (rgrabette2 @ 192.168.1.19, user rasp / pass rasp)
sshpass -p rasp ssh rasp@192.168.1.19

# On device — verify Phase 1 hardware
/usr/sbin/i2cdetect -y 3   # should show 0x36 (distal AS5600)
/usr/sbin/i2cdetect -y 4   # should show 0x36 (proximal AS5600)
/usr/sbin/i2cdetect -y 1   # leftover BMI088 (0x18/0x19/0x68) and audio (0x40)
lsusb -t                    # OAK-D should be on Bus 002 (5000M) — currently on Bus 001 (480M)
vcgencmd get_throttled      # 0x0 = no undervoltage events recorded

# Repo on device
cd /home/rasp/Project/Repo/grabette
git branch --show-current   # should be rgbd
```

## Pointers

- Spectacular AI SDK was evaluated, not adopted (live VIO/SLAM, but adds RPi load and user wasn't satisfied with quick test).
- Luxonis "VSLAM" in DepthAI v3 is a thin wrapper around Basalt + RTAB-Map (early-access). Same upstream as RTAB-Map alone.
- LeRobot v3 has no first-class depth support yet (issue [#1144](https://github.com/huggingface/lerobot/issues/1144) open). Plan: store depth as `dtype: image` with uint16 PNG-on-disk.
- See conversation history for power calculations, sync recommendations, calibration findings.
