# Getting started

This page takes you from nothing to a LeRobot dataset. If you don't have a device yet, start with [mock mode](#try-it-without-hardware) — the whole dashboard works on a laptop.

## Try it without hardware

The device software detects that it isn't running on a Raspberry Pi and falls back to a mock backend, which serves synthetic camera and sensor data. Everything else — the dashboard, sessions, episodes, replay — behaves normally.

You need [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
git clone https://github.com/pollen-robotics/grabette.git
cd grabette
uv sync --package grabette
uv run --package grabette python packages/grabette/main.py
```

Open <http://localhost:8000> and you are in [the dashboard](./dashboard.md).

<Tip warning={true}>

The repository is a single uv **workspace**: a bare `uv sync` builds *every* package, which pulls in gigabytes of PyTorch and MuJoCo. Always pass `--package <name>` unless you deliberately want the full development environment (`uv sync --all-packages`).

</Tip>

## Set up a real Grabette

### 1. Flash and install

Flash **Raspberry Pi OS Lite (64-bit)** onto an SD card with the [Raspberry Pi Imager](https://www.raspberrypi.com/software/), setting a hostname (for example `R-grabette`), a user, your WiFi credentials, and enabling SSH. Both Bookworm and Trixie are tested.

SSH into the Pi and install:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/pollen-robotics/grabette.git
cd grabette/packages/grabette

sudo cp config/config.txt /boot/firmware   # hardware overlays
make install-netdev                        # rights for WiFi scanning
sudo reboot
```

After the reboot, run the one-shot bringup. A Grabette is built as either a **left** or a **right** hand — the angle sensors are mirrored, so the daemon has to be told which one this device is:

```bash
make install-rpi HAND=right    # or HAND=left
make install-systemd           # start on boot
```

`install-rpi` installs the apt-managed `picamera2` stack, the OAK-D udev rule, and a `--system-site-packages` virtualenv against the system Python; `install-systemd` enables the daemon and the Bluetooth WiFi-setup service. Both are idempotent — re-run them if something looks wrong.

<Tip>

If the logs say `Using MockBackend` instead of `RPi hardware detected, using RpiBackend`, the virtualenv didn't pick up the system packages. Re-running `make install-rpi HAND=...` fixes it.

</Tip>

### 2. Get it on the network

If the WiFi you set at flash time isn't the one you need, you can provision the device over Bluetooth — no screen, no SSH. Open the [Bluetooth tool](https://pollen-robotics.github.io/grabette/) in Chrome or Edge, connect to the device, enter the PIN (`00000` by default), scan and pick your network.

### 3. Calibrate

Once, before the first recording: open the gripper so that **both joints are fully extended**, then

```bash
uv run python scripts/calibrate_angles.py
sudo reboot
```

### 4. Record

Browse to `http://<hostname>.local:8000` and record — the physical button on the device and the dashboard's **Episodes** section do the same thing. See [The dashboard](./dashboard.md) for what each section does.

Recordings land in `~/grabette-data/` on the device.

## Turn recordings into a LeRobot dataset

Two routes produce the same thing.

### In the cloud

Upload the episodes to a Hugging Face dataset from the dashboard's **Datasets** section, then run the [Grabette SLAM Space](./spaces.md#grabette-slam--lerobot) on them. Nothing to install; the Space does the SLAM and pushes the LeRobot dataset under your account.

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
