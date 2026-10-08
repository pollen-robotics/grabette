# Build your own

Grabette is open hardware. This page takes you from a box of parts to a device ready for [Getting started](./get_started.md).

## Assembly

Grabette is built from off-the-shelf parts, 3D-printed components and a Raspberry Pi 4.

- 📋 <a href="https://docs.google.com/spreadsheets/d/e/2PACX-1vQ3LyyWI-CiplVPtgrWkmLRYjdDqYhbVJXYt8PNa71FDzbTSMVj1YGV0Zpo5PJeBGJURaz8nZt1_v-8/pubhtml" target="_blank" rel="noopener"><strong>Bill of Materials</strong></a> — the complete parts list, shared by Grabette and Gripette.
- 🧩 <a href="https://cad.onshape.com/documents/0c6175c392788391992ff2ec/w/9f773e5f0eeae1577ae36a05/e/13a89fef2591d863bb0bf186" target="_blank" rel="noopener"><strong>CAD on Onshape</strong></a> — the full assembly.
- 🔩 <a href="https://github.com/pollen-robotics/grabette/blob/develop/packages/grabette/assembly/Grabette_Assembly.pdf" target="_blank" rel="noopener"><strong>Assembly guide</strong></a>, with the matching 3D-print guide in the same folder.

## Flash the Raspberry Pi

Flash **Raspberry Pi OS Lite (64-bit)** onto an SD card with the [Raspberry Pi Imager](https://www.raspberrypi.com/software/), setting a hostname (for example `R-grabette`), the user **`rasp`** (the install targets and the systemd service expect that name), your WiFi credentials, and enabling SSH. Both Bookworm and Trixie are tested.

SSH into the Pi and install:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/pollen-robotics/grabette.git
cd grabette/packages/grabette

sudo cp config/config.txt /boot/firmware   # hardware overlays
make install-audio                         # speaker overlay, must be built before the reboot
make install-netdev                        # rights for WiFi scanning
sudo reboot
```

After the reboot, run the one-shot bringup. A Grabette is built as either a **left** or a **right** hand — the angle sensors are mirrored, so the daemon has to be told which one this device is. It also has to know which depth camera it carries: the Orbbec Gemini 305 (the default) or a Luxonis OAK-D SR:

```bash
make install-rpi HAND=right              # or HAND=left; Gemini 305
make install-rpi HAND=right CAMERA=oakd  # on a Grabette with an OAK-D SR
```

`install-rpi` installs the apt-managed `picamera2` stack, the udev rule for the chosen camera, and a `--system-site-packages` virtualenv against the system Python, and writes `HAND` and `CAMERA` to `/etc/grabette/env`.

<Tip>

If the logs later say `No RPi hardware, using MockBackend` instead of `RPi hardware detected, using RpiBackend`, the virtualenv didn't pick up the system packages. Re-running `make install-rpi HAND=...` fixes it.

</Tip>

## Start the services

```bash
make install-systemd    # start on boot
```

This enables and starts the daemon (`grabette`) and the Bluetooth WiFi-setup service (`grabette-bluetooth`).

Both install targets can be re-run if something looks wrong, with two caveats: re-running `install-rpi` without `CAMERA=oakd` switches an OAK-D device to the Gemini, and `install-systemd` doesn't restart services that are already running — follow it with `sudo systemctl restart grabette grabette-bluetooth`.

## Calibrate the angles

Once, before the first recording: open the gripper so that **both joints are fully extended**, then

```bash
uv run python scripts/calibrate_angles.py
sudo reboot
```

The device is ready: continue with [Getting started](./get_started.md). If the WiFi you set at flash time isn't the one you need, the [Bluetooth tool](./get_started.md#first-time-setup-with-the-bluetooth-tool) changes it.
