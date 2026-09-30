# Getting started

This page takes you from nothing to a LeRobot dataset. If you don't have a device yet, start with [mock mode](#try-it-without-hardware) — the whole dashboard works on a laptop.

## Try it without hardware

The device software detects that it isn't running on a Raspberry Pi and falls back to a mock backend, which serves synthetic camera and sensor data. Everything else — the dashboard, sessions, episodes, replay — behaves normally.

You need [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
git clone https://github.com/pollen-robotics/grabette.git
cd grabette
uv sync --package grabette --extra ui --extra hf   # ui: the dashboard, hf: Hugging Face login
uv run --package grabette python packages/grabette/main.py
```

Open <http://localhost:8000> and you are in [the dashboard](./dashboard.md).

<Tip warning={true}>

The repository is a single uv **workspace**: a bare `uv sync` builds *every* package, which pulls in gigabytes of PyTorch and MuJoCo. Always pass `--package <name>` unless you deliberately want the full development environment (`uv sync --all-packages`).

</Tip>

## Set up a real Grabette

### 1. Flash and install

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
make install-systemd                     # start on boot
```

`install-rpi` installs the apt-managed `picamera2` stack, the udev rule for the chosen camera, and a `--system-site-packages` virtualenv against the system Python, and writes `HAND` and `CAMERA` to `/etc/grabette/env`. `install-systemd` enables the daemon and the Bluetooth WiFi-setup service.

Both can be re-run if something looks wrong, with two caveats: re-running `install-rpi` without `CAMERA=oakd` switches an OAK-D device to the Gemini, and `install-systemd` doesn't restart services that are already running — follow it with `sudo systemctl restart grabette grabette-bluetooth`.

<Tip>

If the logs say `No RPi hardware, using MockBackend` instead of `RPi hardware detected, using RpiBackend`, the virtualenv didn't pick up the system packages. Re-running `make install-rpi HAND=...` fixes it.

</Tip>

### 2. Get it on the network

If the WiFi you set at flash time isn't the one you need, you can provision the device over Bluetooth — no screen, no SSH. Open the [Bluetooth tool](https://pollen-robotics.github.io/grabette/#wifi) in Chrome or Edge on a computer or an Android phone, connect to the device, scan and pick your network. The tool sends the default PIN (`00000`) itself and only asks for it if you changed it.

<Tip warning={true}>

The tool doesn't work on iPhone or iPad, whatever the browser: every iOS browser runs on WebKit, which has no Web Bluetooth.

</Tip>

### 3. Calibrate

Once, before the first recording: open the gripper so that **both joints are fully extended**, then

```bash
uv run python scripts/calibrate_angles.py
sudo reboot
```

### 4. Record

Recording is started and stopped with the **physical button** on the device. For a first try, browse to `http://<hostname>.local:8000` and follow the dashboard's [Test Recording](./dashboard.md#test-recording) page, which walks you through one recording and its replay.

Recordings land in `~/grabette-data/` on the device. For real data — tasks, sessions, several devices at once — sign the device in to Hugging Face and use the [Fleet Space](./spaces.md#grabette-fleet).

## Turn recordings into a LeRobot dataset

Two routes produce the same thing.

### In the cloud

Upload the episodes to a Hugging Face dataset from the [Fleet Space](./spaces.md#grabette-fleet), then build the dataset with the [Grabette SLAM Space](./spaces.md#grabette-slam--lerobot) — the Fleet Space can trigger it for you. Nothing to install; the Space does the SLAM and pushes the LeRobot dataset under your account.

### On your workstation

Requires Docker and Python ≥ 3.12 (LeRobot's minimum). Copy the episodes off the device, then:

```bash
uv sync --package grabette-postprocess
docker pull pollenrobotics/oak-vslam
cd packages/grabette-postprocess

# 1. sanity-check the recordings
uv run python scripts/checks/check_dataset.py -i ~/data/dataset

# 2. expand one episode into the layout the SLAM binary expects
uv run python scripts/pipeline/convert_episode_to_oak.py -i ~/data/dataset/episode

# 3. recover the camera trajectory
uv run python scripts/pipeline/run_oak_slam.py -i ~/data/dataset/episode

# 4. check the trajectory before you build on top of it
uv run python scripts/checks/check_trajectory.py -i ~/data/dataset -v

# 5. build the LeRobot v3 dataset
uv run python scripts/pipeline/generate_dataset.py \
  -i ~/data/dataset \
  --repo_id your-name/your-dataset \
  --task "pick up the cup" \
  --root ~/lerobot_datasets

# 6. push it to the Hub
uv run python scripts/pipeline/push_to_hub.py \
  --repo_id your-name/your-dataset \
  --root ~/lerobot_datasets
```

Steps 2 and 3 take a single episode directory; the others take the dataset directory containing them.

The full pipeline, including the synchronization checks and the Rerun 3D viewer, is documented in the [grabette-postprocess README](https://github.com/pollen-robotics/grabette/tree/develop/packages/grabette-postprocess).

## Next steps

- Train on the result: the repository's [DiffusionPolicy](https://github.com/pollen-robotics/grabette/tree/develop/integrations/DiffusionPolicy) and [π0.5](https://github.com/pollen-robotics/grabette/tree/develop/integrations/Pi05) integrations both consume this dataset format.
- Record with several devices at once — see [the fleet Space](./spaces.md#grabette-fleet).
