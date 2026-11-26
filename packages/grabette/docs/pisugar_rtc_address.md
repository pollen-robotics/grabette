# Moving the PiSugar RTC off 0x68 (BMI088 HAT)

**When you need this:** a grabette with the **older BMI088 IMU HAT** (not the
newer LSM6DSV16X). On that HAT the BMI088 gyro is strapped low at I2C **0x68**,
which is the **same address as the PiSugar 3's DS3231 RTC**. The two devices
wire-AND on the bus, so the gyro is masked and the IMU fails to initialise.
The IMU is best-effort, so capture still starts, but the daemon logs
`No IMU detected on any candidate bus` and episodes carry no `imu_data.json`.

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

- **PiSugar 3 firmware v1.3.8.** The RTC-address register (`0x51`) does not
  exist below this (it reads back `0x00`), and the write-enable (`0x0B`) will
  not latch `0x29`; v1.4.2 has the register but breaks the bus (step 2). Step 1
  checks it; grabettes have shipped with 1.3.4.
- `i2c-tools` on the Pi (`i2cget`/`i2cset`), and I2C bus 1 enabled.
- The IMU/sensor HAT **powered off** and the PiSugar on **wall power** during a
  firmware flash.

> **Note on running the writes.** Claude Code's classifier blocks the flash
> and the raw `i2cset` writes, including over SSH, so run those commands
> yourself (in Claude Code, prefix with `!` to run them in your own shell).
> The read-only checks below are fine to delegate.

---

## Step 0 — confirm the collision

```bash
sudo i2cdetect -y 1
```

On a BMI088 HAT with the RTC still at its default, you see `18` (HAT audio
codec), `19` (BMI088 accel), `57` (PiSugar MCU) and `68`, with **no `33`**.
`0x68` here is the gyro and the RTC answering together. To confirm:

```bash
sudo i2cget -y 1 0x68 0x00   # BMI088 gyro chip id: 0x0F alone; anything else = collision
```

A LSM6DSV16X HAT shows `6a` instead of `19`: stop here, there is nothing to do.

## Step 1 — check the PiSugar firmware version

`pisugar-server` is not installed on grabettes, so read the version straight
from the MCU. It is an ASCII string in registers `0xE2`..`0xEF`, NUL-terminated:

```bash
for r in $(seq 0xe2 0xef); do sudo i2cget -y 1 0x57 $r; done
# 0x76 0x31 0x2e 0x33 0x2e 0x34 0x00 ...  ->  "v1.3.4"
sudo i2cget -y 1 0x57 0x51   # RTC address register: 0x68 on >= 1.3.8, 0x00 below
for r in $(seq 0x52 0x59); do sudo i2cget -y 1 0x57 $r; done   # platform marker
```

Also note the MCU platform marker in `0x52`..`0x59`: all zeros on the original
FM33LC0x firmware, `50 53 49 44 01 01 00 0e` ("PSID", FM33LC0x) once a v1.4.x
image has been flashed, `... 01 02 01 0c` on a CH32L103 board.

- **v1.3.8 on an FM33LC0x board:** skip to step 3.
- **Below 1.3.8, or v1.4.2:** go to step 2. v1.4.2 is *not* fine, see below.
- **CH32L103 board:** this procedure has not been tried on one, and the v1.3.8
  image below is FM33LC0x only.

## Step 2 — flash PiSugar firmware v1.3.8

> ⚠️ **Do not use the official updater (`PiSugarUpdate.sh`) on a BMI088
> grabette.** It only ships v1.4.2 for FM33LC0x boards, and v1.4.2 makes the
> PiSugar MCU ACK **every** address on bus 1: `i2cdetect -y 1` lists
> `0x08`..`0x77`, and an empty address echoes the MCU's last byte. That phantom
> reply is wire-AND'd onto the BMI088 (gyro chip id reads `0x03`, accel `0x02`),
> so the IMU still fails, whatever the RTC address and after a full power
> cycle. Seen on grabette-01 on 2026-11-26; going back to v1.3.8 fixed it.

v1.3.8 is no longer offered by the updater but is still on the PiSugar CDN. The
current programmer (v2.4.5) refuses it (`Missing supported OTA v2 header;
untagged legacy BINs are not accepted`), so flash it with programmer **v2.4.1**,
extracted from its `.deb` rather than installed so the system programmer is
left alone. Programmer v1.6.4 (pulled by old copies of the updater) hangs at
`bootloader not ready`: don't use it either.

Only the application is rewritten, not the bootloader: if v1.3.8 failed to
boot, the official updater can still recover the board from bootloader mode.

With the HAT unplugged and the PiSugar on wall power:

```bash
sudo systemctl stop grabette             # keep the daemon off the I2C bus during the flash
cd ~

curl -fsSL -o pisugar-3-v1.3.8.bin \
  https://cdn.pisugar.com/release/PiSugar3Firmware/fm33lc023n/pisugar-3-application.bin
sha256sum pisugar-3-v1.3.8.bin
# c1e18e9d4e3c29b7d1c765f4f74edabb76136ec85944482abc98cf2021414c9e

wget https://github.com/PiSugar/pisugar-power-manager-rs/releases/download/v2.4.1/pisugar-programmer_2.4.1-1_arm64.deb
dpkg-deb -x pisugar-programmer_2.4.1-1_arm64.deb p241
p241/usr/bin/pisugar-programmer --version          # pisugar-programmer 2.4.1
echo y | sudo p241/usr/bin/pisugar-programmer -r pisugar-3-v1.3.8.bin
```

If it stalls at `bootloader not ready`, press the PiSugar's reset button. The
programmer panics with `No such file or directory` when the `.bin` path is
wrong. Then rerun step 1 (expect `v1.3.8`, `0x51` = `0x68`) and check, still
without the HAT, that `sudo i2cdetect -y 1` shows **only** `57` and `68`.

## Step 3 — re-address the RTC to 0x33

All writes go to the **PiSugar MCU at `0x57`** (the power-management chip) — **not**
to `0x68`. The RTC address lives in MCU register `0x51`, gated by the
write-enable register `0x0B`.

**Address encoding for register `0x51`:** take the new 7-bit address `N`. If `N`
has **even parity**, write it raw; if **odd parity**, set the high bit
(`N | 0x80`). `0x33` has even parity, so it is written as `0x33`; `0x68` (the
default, to revert) has odd parity, so it is written as `0xE8`. A value with the
wrong parity is silently ignored: the register keeps its old value.

```bash
# bus 1, MCU at 0x57
sudo i2cset -y 1 0x57 0x0B 0x29      # unlock write-enable
sudo i2cset -y 1 0x57 0x51 0x33      # set RTC address = 0x33 (even parity -> raw)
sudo i2cset -y 1 0x57 0x0B 0x00      # re-lock
sleep 3                              # MCU needs ~3s to apply
sudo i2cget -y 1 0x57 0x51           # read back -> expect 0x33
```

## Step 4 — verify

Power the Pi off, plug the HAT back in, boot, then:

```bash
sudo i2cdetect -y 1
# Expect: 18 (codec), 19 (BMI088 accel), 33 (RTC, moved), 57 (PiSugar MCU), 68 (BMI088 gyro).
sudo i2cget -y 1 0x68 0x00   # -> 0x0F, the gyro alone
sudo systemctl restart grabette
journalctl -u grabette | grep -i imu   # -> "Detected bmi088 IMU on I2C bus 1"
```

The matching config is already in the repo:

- `grabette/config.py` → `DEFAULT_IMU_CANDIDATES` includes
  `("bmi088", 1, {"gyro_addr": 0x68})`
- `imu_hz = 200` (BMI088 poll == ODR == 200 Hz; the 240 Hz LSM6 target would
  round the BMI088 to a 400 Hz ODR that reads empty accel on this bus).

The move is permanent per device — you only do this once per BMI088 grabette.
