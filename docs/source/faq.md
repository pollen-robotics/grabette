# FAQ

<Tip warning={true}>

This page is a starting point, seeded from the questions the repository's own documentation already answers. If you hit something that isn't here, [open an issue](https://github.com/pollen-robotics/grabette/issues) — that's how this page grows.

</Tip>

## Do I need a robot to collect data?

No. You demonstrate the task with your hand, holding the Grabette. The recording contains the camera trajectory and the finger-joint angles, and carries no assumption about which arm will eventually replay it. A robot only enters the picture at training and deployment time.

## Do I need the OAK-D camera?

On **Grabette**, yes. Trajectory recovery is visual-inertial and uses the OAK-D SR's depth and IMU streams; without it there is no SLAM and therefore no dataset. It is off by default to save battery, so turn it on before recording.

On **Gripette** the OAK-D is optional — the standard motor-and-camera service doesn't need it.

## The daemon says `Using MockBackend` on a real Raspberry Pi

The virtualenv didn't pick up the apt-managed `picamera2` and `libcamera` packages, so the hardware backend can't import and the daemon falls back to mock data. Re-run `make install-rpi HAND=...`; it rebuilds the venv with `--system-site-packages` against the system Python and verifies every import.

## Why does `make install-rpi` insist on `HAND=`?

A device is built as a left or a right hand, and the angle sensors are mounted mirrored between the two. The daemon has to know which it is to interpret them. The value is written to `/etc/grabette/env` and persists across reboots.

## `uv sync` is downloading gigabytes of PyTorch

You ran it without `--package`. The repository is a single uv workspace, so a bare `uv sync` from anywhere in it resolves *every* package, training integrations included. Use `uv sync --package grabette` for one package, or `uv sync --all-packages` when you genuinely want the full development environment.

## The `.stl` files are 130-byte text files

The repository stores mesh assets in Git LFS and you cloned before running `git lfs install`. Install it, then `git lfs pull` in the clone. If you're deploying to a Pi, which never loads the meshes, skip them entirely with `GIT_LFS_SKIP_SMUDGE=1 git clone ...`.

## I can't reach `http://<hostname>.local:8000`

`.local` resolution needs mDNS, which some networks and some corporate laptops block. Use the device's IP address instead. If the device isn't on the network at all, provision WiFi over Bluetooth with the [BT tool](https://pollen-robotics.github.io/grabette/).

## The Bluetooth tool won't connect

It relies on Web Bluetooth, so it needs Chrome or Edge — Safari and Firefox won't work. Chrome may also need `chrome://flags/#enable-experimental-web-platform-features` enabled. If a connection hangs at pairing, clear the stale bond on both sides: `bluetoothctl remove <device-mac>` on the Pi, and *Forget* the device in `chrome://bluetooth-internals`.

## Dataset generation fails on Python 3.11

LeRobot 0.6 requires Python ≥ 3.12, so dataset generation, publishing and visualization do too. The device packages stay installable on 3.11 — the requirement is gated by environment markers — but the post-processing side needs the newer interpreter.

## Recordings from two devices don't line up

Synchronized group recording works by having every device wait out a shared UTC instant *on its own clock*, so a clock offset between devices becomes an offset between recordings. `make install-ntp`, which `install-rpi` runs for you, pins all devices to a single coordinated time service instead of the default pool, where each device would otherwise land on a different physical server. Check it with `timedatectl timesync-status`. For sub-millisecond sync, run an NTP server on the LAN and point the devices at it.

## Can I use Grabette with my own robot?

Yes — that's the design. The core packages carry no dependency on any particular arm; the OpenArm integration in the repository is a worked example, not a requirement. To target another platform, add your own integration alongside it.
