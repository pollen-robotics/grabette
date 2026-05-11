# Grabette V2 (rgbd branch) — status & resume guide

Working notes for the V2 hardware migration. Pick this up to continue without losing context.

## Goal

Migrate from V1 (Grove HAT + bit-banged I2C + BMI088 IMU) to V2 (custom HAT + hardware I2C + OAK-D SR over USB3). The OAK-D SR replaces the BMI088 and adds RGBD streams. The RPi fisheye camera, AS5600 angle sensors, and a switch/LED stay.

## Hardware summary (V2)

| Function | Bus / interface | Pins | Address | Notes |
|---|---|---|---|---|
| RPi camera (fisheye) | CSI | — | — | OV5647, 1296x972 @ 50fps, unchanged from V1 |
| AS5600L distal | hw I2C3 | GPIO 4 SDA / 5 SCL | 0x40 | `dtoverlay=i2c3,pins_4_5` |
| AS5600L proximal | hw I2C4 | GPIO 8 SDA / 9 SCL | 0x40 | `dtoverlay=i2c4,pins_8_9` |
| Switch | direct GPIO | GPIO 10 | — | input, active-low, internal pull-up (some bounce on press, see notes) |
| LED | direct GPIO | GPIO 11 | — | output, **active-low** (inverted from V1 — `gpioset 11=inactive` lights the LED on the wire) |
| OAK-D SR | USB3 | — | — | stereo CAM_B + CAM_C + BNO086 IMU (9-axis w/ fusion). Bootloader endpoint stays USB 2.0; streaming endpoint at USB 3.0 SuperSpeed (use `device.getUsbSpeed()` as authoritative). |
| BMI088 (on HAT, unused) | hw I2C1 | GPIO 2/3 | 0x18, 0x19, 0x68 | leftover on HAT, ignore |
| Audio codec (on HAT, unused) | hw I2C1 | GPIO 2/3 | 0x40 | not used in software |

Power: device currently runs from a USB-C wall supply. PiSugar 3+ for V1 prototypes. A custom 2S battery pack with dual buck regulators is planned (Pi rail + OAK-D rail, common ground).

## Phase 1 — DONE

Mechanical changes: hardware I2C overlays, drop BMI088, switch/LED on direct GPIO. All committed to `rgbd` branch (commit `f4872ab` "wip" or later).

| Change | Files |
|---|---|
| Hardware I2C overlays for AS5600Ls | `config/config.txt` |
| Default I2C bus numbers 4/5 → 3/4 | `grabette/hardware/angle.py` |
| AS5600 address 0x36 → 0x40 (chips are AS5600L variant) | `grabette/hardware/angle.py` |
| LedButton default pins 22/23 → 11/10 | `grabette/hardware/button.py` |
| LedButton LED is now active-low (V2 wiring inverted from V1) — gpiod `active_low=True` on the line | `grabette/hardware/button.py` |
| Drop BMI088 path | `grabette/backend/rpi.py` (rewrite), deleted `grabette/hardware/imu.py` and `grabette/hardware/bmi088.py` |
| Save `frame_timestamps.json` and `angle_data.json` per episode (no more `imu_data.json` from rpi backend) | `grabette/backend/rpi.py` |

### Deployed and verified on device (rgrabette2 @ 192.168.1.19)

- `/dev/i2c-3` and `/dev/i2c-4` come up correctly with `i2c_bcm2835` driver after deploying new `config.txt` and rebooting.
- **AS5600L distal at i2c-3 / 0x40**: detected. Raw angle reads sensible 12-bit values that change with magnet rotation. STATUS shows `MD=1` (magnet detected) but `ML=1` (too weak) and AGC saturated at 0x7F — **magnet too far from chip face or wrong magnet**, mechanical issue.
- **AS5600L proximal at i2c-4 / 0x40**: same as above.
- **Switch GPIO 10**: works with `gpiomon -c gpiochip0 --bias=pull-up 10`. Falling edge on press, rising on release. Some contact bounce visible (multiple falling events per press) — `LedButton.wait_for_press()` polling-based API is debounce-safe; an event-driven user would need explicit debouncing.
- **LED GPIO 11**: works. Confirmed active-low — fix applied to `button.py` using gpiod `active_low=True` so `led_on()` still means "LED illuminated" on both V1 and V2 boards.
- `i2c-1` shows leftover BMI088 (`0x18/0x19/0x68`) and HAT audio codec (`0x40`) — unused by software.
- OAK-D recognized on USB but at **480 Mbps (USB 2.0)** — see open issues.

### Known downstream breakage (acceptable on this branch, fix in Phase 2)

- `grabette/replay.py`, `grabette/app/routers/replay.py` still hard-require `imu_data.json` — replay tool will fail on V2 episodes until updated.
- `grabette/hf.py`, `grabette/session.py` reference `imu_data.json` in comments/docstrings; runtime falls back gracefully.
- `grabette/backend/mock.py` still writes fake `imu_data.json` — fine, it's a dev tool.

## Resolved — sensors are AS5600L, address is 0x40

The "no AS5600 at 0x36" investigation revealed the chips on the HAT are **AS5600L** (not AS5600). Same register layout but default I2C address is `0x40`. After isolating one sensor on i2c-3 only and seeing `0x40` appear with `0x40` empty on i2c-4, the variant was confirmed by the electronics designer.

Resolution: `AS5600_ADDRESS = 0x40` in `grabette/hardware/angle.py`. No hardware change. Pull-ups, power, and wiring all turned out to be fine.

Notes for future:
- The AS5600L supports a **user-programmable I2C address** via register `0x20` (OTP-burnable) or `0x21` (volatile). This would allow both sensors on one I2C bus and free up the second bus. Considered as "Option B" but not adopted — the dual-bus setup works and avoids OTP burns. Worth revisiting if a HAT revision wants cleaner cable management.
- OTP burn details: write `0x40` to register `0xFF` to execute BURN_SETTING (programs MANG, CONFIG, I2CADDR). Bit-by-bit one-way (0→1 only), so from default `0x40` you can burn to e.g. `0x41`, `0x44`, `0x60`, etc. but not back to `0x36`.

## Resolved — OAK-D USB 2.0 was the bootloader endpoint

`lsusb -t` showing the device on Bus 001 (480 Mbps) is **expected behavior** for OAK-D devices. Quoting the Luxonis docs:

> Initial boot phase: "For showing up when plugged in. We use this endpoint to load the firmware onto the device, which is a usb-boot technique. This device is USB2."
> Runtime phase: "For running the actual code. This shows up after USB booting and is USB3."

After `depthai.Device()` connects, `device.getUsbSpeed()` returns `UsbSpeed.SUPER` (USB 3.0). The bootloader interface remains visible in `lsusb -t` on Bus 001; the SuperSpeed data path goes through Bus 002 but isn't exposed the same way to lsusb. Use `device.getUsbSpeed()` to verify.

Confirmed working on the battery-powered device.

## Open issue — magnets too weak

Both AS5600L sensors report `MD=1` (magnet detected), `ML=1` (too weak), AGC saturated at 0x7F (max for 3.3V mode). Software side is correct; this is a mechanical assembly problem:
- Magnet may be too far from the chip face (should be ~0.5–3 mm air gap)
- Magnet may not be diametrically magnetized (AS5600L needs a diametric magnet, not axial)
- Magnet grade may be too weak — try a stronger neodymium ring/disk

No software impact yet. Will need to fix before any meaningful angle data can be captured.

## Phase 2 — IN PROGRESS

### `grabette/hardware/oakd.py` — DONE (v2, H.264)

`OakdCapture` class wrapping depthai v3. Pipeline = 2× Camera (CAM_B/CAM_C @ NV12) → 2× VideoEncoder (H.264, 8 Mbps, no B-frames, keyframe every 30) → host queues; plus IMU node streaming accel/gyro/rotation_vector at 200 Hz. Writer threads dump:
- left/right H.264 elementary streams → muxed to `.mp4` on stop (uses ffmpeg, same approach as `VideoCapture`)
- IMU samples → `oakd_imu.json` (flat list with `kind`, `device_us`, `host_ms`, `value`)
- Per-frame timestamps → `oakd_left_timestamps.json` / `oakd_right_timestamps.json` (independent of mp4 container fps metadata)
- Factory calibration → `oakd_calib.json` (handler.eepromToJson())
- First per-stream device↔host clock pair → `oakd_clock_pairs.json` (offline tools fit drift correction from this; add more pairs in v3 if drift becomes an issue)

Verified end-to-end on device: 3 s capture = 88+89 frames at 30 fps, ~6 MB total, valid mp4 playable in ffprobe. ~32× smaller than the v1 PNG-sequence prototype.

Open follow-ups (not blockers):
- Depth stream not yet captured. To add: `StereoDepth` node fed from both cameras; depth output → uint16 PNG sequence in `oakd_depth/` (no codec preserves uint16 well).
- Periodic clock pairs (currently only the first pair per stream is logged). Add `(device_us, host_ms)` every N frames if offline tools want drift correction beyond a single offset.
- Bitrate (8 Mbps) and keyframe frequency (every 30 frames) are reasonable defaults; tune later if needed.

### Still TODO

- **Sync update**: keep `SyncManager` (monotonic) as master; OAK-D timestamps stored as `{device_us, host_ms}` pairs; offline tools fit linear (drift) correction.
- **`grabette/backend/rpi.py`** — wire `OakdCapture` into the capture session. Update `start_capture` / `stop_capture` to drive it alongside the RPi camera and angle sensors.
- **Output schema** per episode (current — as produced by `OakdCapture` standalone; depth still pending):
  ```
  raw_video.mp4                  RPi fisheye (unchanged)
  metadata.json                  capture summary
  frame_timestamps.json          RPi camera frame ts (Phase 1)
  angle_data.json                proximal+distal (Phase 1)
  oakd_left.mp4                  1280x800 mono left, H.264 ~8 Mbps
  oakd_right.mp4                 1280x800 mono right, H.264 ~8 Mbps
  oakd_left_timestamps.json      per-frame device_us + host_ms
  oakd_right_timestamps.json     same shape
  oakd_imu.json                  accel/gyro/rotation_vector, ~200Hz, kind+device_us+host_ms+value
  oakd_calib.json                factory intrinsics + extrinsics (eepromToJson dump)
  oakd_clock_pairs.json          first device_us ↔ host_ms pair per stream
  # TODO when depth lands:
  oakd_depth/                    uint16 PNG sequence
  oakd_depth_timestamps.json     per-frame ts
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

1. **Connect the better PSU** (official Pi 5V/3A or equivalent). User noted current PSU may be marginal and is suspected of causing the OAK-D USB 2.0 fallback.
2. **Re-verify OAK-D on USB 3.0** — with proper PSU, plug OAK-D into a blue port with a known USB 3 cable. `lsusb -t` should now show the device on Bus 002 at 5000M. If still 480M, swap the cable.
3. **Magnets (optional, for full angle capture test)** — adjust mechanical assembly so AGC drops below 0x7F and `ML` clears. Until fixed, raw angle values are noisy / unreliable but the chip is detected and software path works.
4. **End-to-end `AngleCapture` smoke test** — run a small Python script using `grabette.hardware.AngleCapture` reading both sensors simultaneously, log values, verify they change with rotation.
5. **End-to-end `LedButton` smoke test** — instantiate `LedButton()`, call `led_on()`, `led_off()`, `led_blink()`, `wait_for_press()`. With the v2 polarity fix, `led_on()` should illuminate the LED on the wire.
6. **Then Phase 2** — start with `grabette/hardware/oakd.py` (see Phase 2 TODO).

## Quick commands

```bash
# Local dev machine — repo
cd /home/steve/Project/Repo/GRABETTE/grabette
git checkout rgbd
git status

# Device (rgrabette2 @ 192.168.1.19, user rasp / pass rasp)
sshpass -p rasp ssh rasp@192.168.1.19

# On device — verify Phase 1 hardware
sudo i2cdetect -y 3                                          # should show 0x40 (distal AS5600L) ✓
sudo i2cdetect -y 4                                          # should show 0x40 (proximal AS5600L) ✓
sudo i2cdetect -y 1                                          # BMI088 (0x18/0x19/0x68) + HAT audio codec (0x40), unused
sudo i2cget -y 3 0x40 0x0B                                   # distal STATUS (0x67 = MD/ML set → magnet too weak)
sudo i2cget -y 3 0x40 0x1A                                   # distal AGC (0x7F = saturated → magnet too far)
sudo gpiomon -c gpiochip0 --bias=pull-up 10                  # switch press = falling, release = rising ✓
sudo gpioset -c gpiochip0 11=active                          # LED on (after the active_low=True fix in code) ✓
lsusb -t                                                     # OAK-D should be on Bus 002 (5000M) — currently on Bus 001 (480M)
vcgencmd get_throttled                                       # 0x0 = no undervoltage events recorded

# Repo on device
cd /home/rasp/Project/Repo/grabette
git branch --show-current   # should be rgbd
```

## Pointers

- Spectacular AI SDK was evaluated, not adopted (live VIO/SLAM, but adds RPi load and user wasn't satisfied with quick test).
- Luxonis "VSLAM" in DepthAI v3 is a thin wrapper around Basalt + RTAB-Map (early-access). Same upstream as RTAB-Map alone.
- LeRobot v3 has no first-class depth support yet (issue [#1144](https://github.com/huggingface/lerobot/issues/1144) open). Plan: store depth as `dtype: image` with uint16 PNG-on-disk.
- See conversation history for power calculations, sync recommendations, calibration findings.
