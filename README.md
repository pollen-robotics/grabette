
# GRABETTE

[![Discord](https://img.shields.io/badge/Discord-Join_the_Community-7289DA?logo=discord&logoColor=white)](https://discord.gg/Y7FgMqHsub)

<img align="left" width="20%" src="docs/images/grabette_logo_small.png">
<br><br><br>
Open-source toolkit for collecting robotic manipulation demonstrations and
turning them into training-ready datasets.

A GRABETTE rig records synchronized **camera + depth** streams (+ IMU with the OAK-D SR) from hand-held or
gripper-mounted devices, recovers camera trajectories with SLAM, and exports
[LeRobot](https://huggingface.co/docs/lerobot) datasets for policy learning.
The data-collection pipeline is **robot-agnostic**.

🔩 **Build the hardware:** [Bill of Materials](https://docs.google.com/spreadsheets/d/e/2PACX-1vQ3LyyWI-CiplVPtgrWkmLRYjdDqYhbVJXYt8PNa71FDzbTSMVj1YGV0Zpo5PJeBGJURaz8nZt1_v-8/pubhtml) · [CAD — Onshape](https://cad.onshape.com/documents/0c6175c392788391992ff2ec/w/9f773e5f0eeae1577ae36a05/e/13a89fef2591d863bb0bf186) · Assembly guides: [Grabette](packages/grabette/assembly/) · [Gripette](packages/gripette/assembly/)
<br><br><br>

https://github.com/user-attachments/assets/6db9dd7b-1762-4004-8a76-ce76323499ba

<br><br><br>
## Components

### `packages/` — the devices (what runs on the Raspberry Pi images)

| Package | Role | Target | Interface |
|---|---|---|---|
| [`grabette`](packages/grabette) | Hand held data-collection device | Raspberry Pi | HTTP/WebSocket, :8000 |
| [`gripette`](packages/gripette) | Robot mounted Gripper motor | Raspberry Pi Zero 2W | gRPC, :50051 |
| [`casquette (WIP)`](packages/casquette) | POV head-mounted device | Raspberry Pi Zero 2W | HTTP/WebSocket, :8001 |

### Related repositories

| Repository | Role |
|---|---|
| [grabette-slam](https://github.com/pollen-robotics/grabette-slam) | Raw recordings → SLAM → LeRobot dataset: the `grabette-postprocess` package and the Hugging Face Space that runs it |
| [grabette-policy](https://github.com/pollen-robotics/grabette-policy) | Policy learning: Diffusion Policy and π0.5 training, action representations (grasp projection, chunk-relative), diagnostics, and the OpenArm + Gripette robot (model, sim, real driver, teleop bridge) |
| [grabette-os](https://github.com/pollen-robotics/grabette-os) | The `grabetteos` / `gripetteos` Raspberry Pi images |

**Using GRABETTE with a different robot arm:**
the devices carry no OpenArm dependency. The OpenArm integration in
[grabette-policy](https://github.com/pollen-robotics/grabette-policy) is the
reference example for adding another platform.


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
uv sync --all-packages          # full workspace dev environment
uv run --package grabette python packages/grabette/main.py   # run a service (mock backend by default)
```

> **The one rule to know:** this repo is a single uv **workspace** — one shared
> `.venv` and one `uv.lock` at the root. A bare `uv sync`, run from the repo
> root, builds the **whole workspace** and installs every package's
> dependencies. Therefore:
>
> - **Single package / deployment** → `uv sync --package <name>`
>   (extras attach to it: `uv sync --package grabette --extra rpi`).
> - **Full dev environment** → `uv sync --all-packages`.

**On-device install (Raspberry Pi):** the simplest path is to flash the
prebuilt `grabetteos` / `gripetteos` image from the
[latest grabette-os release](https://github.com/pollen-robotics/grabette-os/releases/latest),
which has everything below baked in. For a manual install, each device package ships a `make
install-rpi` target that builds the `--system-site-packages` venv `picamera2`
needs (a bare `uv sync` skips it and the service falls back to the mock
backend). Follow the package's own README for the exact steps:
[grabette](packages/grabette), [gripette](packages/gripette),
[casquette](packages/casquette) *(WIP)*.

## License

Apache-2.0. See [LICENSE](LICENSE).
