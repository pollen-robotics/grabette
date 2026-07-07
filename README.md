
# GRABETTE
<img align="left" width="50%" src="docs/images/grabette_logo_small.png">
Open-source toolkit for collecting robotic manipulation demonstrations and
turning them into training-ready datasets.

A GRABETTE rig records synchronized **camera + IMU** streams from hand-held or
gripper-mounted devices, recovers camera trajectories with SLAM, and exports
[LeRobot](https://huggingface.co/docs/lerobot) datasets for policy learning.
The data-collection pipeline is **robot-agnostic**.

## Components

### `packages/` — robot-agnostic core

| Package | Role | Target | Interface |
|---|---|---|---|
| [`grabette`](packages/grabette) | Hand held data-collection device | Raspberry Pi | HTTP/WebSocket, :8000 |
| [`gripette`](packages/gripette) | Robot mounted Gripper motor | Raspberry Pi Zero 2W | gRPC, :50051 |
| [`grabette-postprocess`](packages/grabette-postprocess) | Data postprocess + SLAM → LeRobot dataset generation | Workstation | CLI |
| [`casquette (WIP)`](packages/casquette) | POV head-mounted device | Raspberry Pi Zero 2W | HTTP/WebSocket, :8001 |

### `integrations/` — integration example (OpenArm 7-DOF arm + Gripette)

| Package | Role |
|---|---|
| [`openarm_gripette`](integrations/openarm/openarm_gripette) | Code to control the OpenArm + Gripette robot |
| [`openarm_gripette_simu`](integrations/openarm/openarm_gripette_simu) | MuJoCo simulation of OpenArm + Gripette and synthetic data collection |
| [`openarm_gripette_model`](integrations/openarm/openarm_gripette_model) | Robot description (URDF / MuJoCo XML) and mesh assets, generated from Onshape |
| [`DiffusionPolicy`](integrations/DiffusionPolicy) | Diffusion Policy training code |

**Using GRABETTE with a different robot arm:**
the core in `packages/` carries no
OpenArm dependency. To target another platform, add an
`integrations/<your-arm>/` alongside `openarm/` — the OpenArm integration is the
reference example.


## Cloning

The repo uses **Git LFS** for mesh assets (`*.stl`, see `.gitattributes`). Install LFS once per workstation, then clone normally:

```bash
sudo apt install git-lfs             # Debian / Ubuntu / Pi OS — install the binary first
                                     # (macOS: brew install git-lfs; see git-lfs.com for others)
git lfs install                      # one-time per user — configures git filters
git clone git@github.com:pollen-robotics/grabette.git
```

If you cloned **before** `git lfs install`, the `.stl` files are 130-byte pointer text files. Fetch the real binaries:
```bash
cd grabette
git lfs pull
```

Verify:
```bash
file packages/grabette/urdf/grabette_right/assets/*.stl | head -3
# expected: "Binary"   |   bad: "ASCII text" (pointer file → run `git lfs pull`)
```

For on-device installs where you don't need the meshes (Pi services don't load them), skip LFS to save disk + bandwidth:
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:pollen-robotics/grabette.git
```

## Development

Requires [uv](https://docs.astral.sh/uv/). Python ≥ 3.11.

```bash
uv sync --all-packages          # build the full workspace environment
uv run --package grabette python packages/grabette/main.py   # run a service (mock backend by default)
```

Work on a single component without pulling the rest:

```bash
uv sync --package grabette-postprocess
uv run --package grabette-postprocess python scripts/arducam_slam/generate_dataset.py --help
```

> **The one rule to know:** this repo is a single uv **workspace** — one shared
> `.venv` and one `uv.lock` at the root. A bare `uv sync`, run from *anywhere*
> in the repo (including inside a package directory), operates on the **whole
> workspace** and installs every package's dependencies — gigabytes of
> torch/mujoco on a Raspberry Pi if you're not careful. Therefore:
>
> - **Deployment / single package** → always `uv sync --package <name>`
>   (extras attach to it: `uv sync --package grabette --extra rpi`).
> - **Full dev environment** → `uv sync --all-packages`.
> - `uv run --package <name> …` runs against that package's dependency set.
> - Exception: `integrations/DiffusionPolicy` is deliberately **standalone**
>   (own `uv.lock`, heavy training pins) — inside it, a plain `uv sync` is
>   correct and touches nothing else.

### Notes

- **Python / `lerobot`:** `lerobot` (used by `grabette-postprocess` and the sim's
  `dataset` extra) requires Python ≥ 3.12, so it is gated by an environment
  marker. The on-device services still install and run on Python 3.11
  (Raspberry Pi OS Bookworm) — they don't depend on `lerobot`.
- **OpenArm sim system deps:** `placo` (sim kinematics) dynamically links
  `liburdfdom`; install it from your distro's packages before running the sim.
- **Raspberry Pi install:** use the `make install-rpi` target (see below) — a
  bare `uv sync` skips the apt deps and `--system-site-packages` venv that
  `picamera2` needs, and the service silently falls back to the mock backend.

## Running on a Raspberry Pi device

The on-device services (`grabette`, `casquette`, `gripette`) install through a
`make` target that builds the `--system-site-packages` venv at the workspace
root, with the device's apt-provided `picamera2`/`libcamera`:

```bash
# clone skipping the meshes (on-device services don't load them)
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:pollen-robotics/grabette.git
cd grabette/packages/grabette

make install-rpi HAND=right                    # or HAND=left — grabette is built mirrored per side
uv run --package grabette python -m grabette   # run the service (auto-detects hardware)
make install-systemd                           # installs BOTH grabette.service AND grabette-bluetooth.service
```

On success the log shows `RPi hardware detected, using RpiBackend` (not
`MockBackend`). `casquette` follows the same pattern from `packages/casquette` (no `HAND=`,
it's hand-agnostic); `gripette` likewise needs `HAND=` from `packages/gripette`. Run `make help`
in any package for the available targets.

## License

Apache-2.0. See [LICENSE](LICENSE).
