# Moving the PiSugar RTC off 0x68 (BMI088 HAT)

**When you need this:** a grabette with the **older BMI088 IMU HAT** (not the
newer LSM6DSV16X). On that HAT the BMI088 gyro is strapped low at I2C **0x68**,
which is the **same address as the PiSugar 3's DS3231 RTC**. The two devices
wire-AND on the bus, so the gyro is masked and the IMU fails to initialise —
and because recording requires a working IMU, capture refuses to start.

A LSM6DSV16X grabette does **not** need any of this (that sensor sits at 0x6A).

**The fix:** re-address the PiSugar RTC from `0x68` to `0x33`. The change is
stored in the PiSugar MCU and **persists across reboots**; the OS keeps time
over NTP, so nothing needs the RTC at 0x68 afterwards. Once moved, `0x68`
answers as the BMI088 gyro (chip id `0x0F`) and `detect_imu_capture` finds it.

> Symptom that confirms the collision beforehand: reading the RTC seconds shows
> the ones digit wrapping 0–9 but the tens digit never advances (the gyro's
> low byte is being wire-AND'd onto the RTC register).

---

## Prerequisites

- **PiSugar 3 firmware ≥ 1.3.8.** The RTC-address register (`0x51`) does not
  exist below this, and the write-enable (`0x0B`) will not latch `0x29`. Check
  your version in the PiSugar web UI or app; flash first if it is older
  (right-grabette shipped 1.3.4 and had to be flashed — see below).
- `i2c-tools` on the Pi (`i2cget`/`i2cset`), and I2C bus 1 enabled.
- The IMU/sensor HAT **powered off** and the PiSugar on **wall power** during a
  firmware flash.

> **Note on running the writes.** Claude Code's write classifier blocks raw
> `i2cset`/flash commands, so run the commands in this doc yourself (in Claude
> Code, prefix with `!` to run them in your own shell).

---

## Step 1 — (only if fw < 1.3.8) flash the PiSugar firmware

The official updater installs a broken programmer for this firmware range:

```bash
# DON'T rely on this alone — it pulls programmer v1.6.4, which HANGS at
# "bootloader not ready" on fw >= 1.2.4 (it fails to unlock 0x0B before reset).
curl https://cdn.pisugar.com/release/PiSugarUpdate.sh | bash
```

Use programmer **v2.4.1** instead:

```bash
# Install the working programmer
wget https://github.com/PiSugar/pisugar-power-manager-rs/releases/download/v2.4.1/pisugar-programmer_2.4.1-1_arm64.deb
sudo dpkg -i pisugar-programmer_2.4.1-1_arm64.deb

# Flash the application firmware (HAT OFF, PiSugar on wall power)
echo y | pisugar-programmer -r pisugar-3-application.bin
```

(Download the matching `pisugar-3-application.bin` for ≥ 1.3.8 from the PiSugar
releases.) After flashing, confirm the version before continuing.

## Step 2 — re-address the RTC to 0x33

All writes go to the **PiSugar MCU at `0x57`** (the power-management chip) — **not**
to `0x68`. The RTC address lives in MCU register `0x51`, gated by the
write-enable register `0x0B`.

**Address encoding for register `0x51`:** take the new 7-bit address `N`. If `N`
has **even parity**, write it raw; if **odd parity**, set the high bit
(`N | 0x80`). `0x33` has even parity, so it is written as `0x33`.

```bash
# bus 1, MCU at 0x57
sudo i2cset -y 1 0x57 0x0B 0x29      # unlock write-enable
sudo i2cset -y 1 0x57 0x51 0x33      # set RTC address = 0x33 (even parity -> raw)
sudo i2cset -y 1 0x57 0x0B 0x00      # re-lock
sleep 3                              # MCU needs ~3s to apply
sudo i2cget -y 1 0x57 0x51           # read back -> expect 0x33
```

## Step 3 — verify

```bash
sudo i2cdetect -y 1
# 0x68 should now be the BMI088 gyro (no longer the RTC at 0x68).
# Expect to also see: 0x19 (BMI088 accel), 0x57 (PiSugar MCU), 0x33 (RTC, moved).
```

Then let the grabette service probe it — a good capture logs
`Detected bmi088 ... on I2C bus 1`. The matching config is already in the repo:

- `grabette/config.py` → `DEFAULT_IMU_CANDIDATES` includes
  `("bmi088", 1, {"gyro_addr": 0x68})`
- `imu_hz = 200` (BMI088 poll == ODR == 200 Hz; the 240 Hz LSM6 target would
  round the BMI088 to a 400 Hz ODR that reads empty accel on this bus).

The move is permanent per device — you only do this once per BMI088 grabette.
