# GRABETTE

Open-source toolkit for collecting robotic manipulation demonstrations and
turning them into training-ready datasets.

A GRABETTE rig records synchronized **camera + IMU** streams from hand-held or
gripper-mounted devices, recovers camera trajectories with SLAM, and exports
[LeRobot](https://huggingface.co/docs/lerobot) datasets for policy learning.
The data-collection pipeline is **arm-agnostic** — OpenArm is provided as one
worked integration, not a requirement.

> This is a uv **workspace monorepo**. It supersedes the former per-component
> repositories (`grabette`, `casquette`, `gripette`, `grabette-data`,
> `openarm_gripette_simu`, `openarm_gripette_model`), which are archived
> read-only. Electronics and screen firmware live in a separate hardware repo.

## Components

### `packages/` — arm-agnostic core

| Package | Role | Target | Interface |
|---|---|---|---|
| [`grabette`](packages/grabette) | On-device data-collection service (camera + IMU + angle/OAK-D) | Raspberry Pi | HTTP/WebSocket, :8000 |
| [`casquette`](packages/casquette) | POV head-mounted camera + IMU collection service | Raspberry Pi Zero 2W | HTTP/WebSocket, :8001 |
| [`gripette`](packages/gripette) | Gripper motor + camera service | Raspberry Pi Zero 2W | gRPC, :50051 |
| [`grabette-postprocess`](packages/grabette-postprocess) | SLAM/VIO (OAK-D + RTAB-Map, Dockerized) → LeRobot dataset generation | Workstation | CLI |

### `integrations/openarm/` — reference integration (OpenArm 7-DOF arm)

| Package | Role |
|---|---|
| [`openarm_gripette_simu`](integrations/openarm/openarm_gripette_simu) | MuJoCo simulation of OpenArm + Gripette, with gRPC gripper (:50051) and arm (:50052) control and synthetic data collection |
| [`openarm_gripette_model`](integrations/openarm/openarm_gripette_model) | Robot description (URDF / MuJoCo XML) and mesh assets, generated from Onshape |

**Using GRABETTE with a different arm:** the core in `packages/` carries no
OpenArm dependency. To target another platform, add an
`integrations/<your-arm>/` alongside `openarm/` — the OpenArm integration is the
reference example.

## Layout

```
packages/                       arm-agnostic core (uv workspace members)
integrations/openarm/           reference integration for the OpenArm arm
pyproject.toml                  uv workspace root
uv.lock                         single lock for the whole workspace
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

### Notes

- **Python / `lerobot`:** `lerobot` (used by `grabette-postprocess` and the sim's
  `dataset` extra) requires Python ≥ 3.12, so it is gated by an environment
  marker. The on-device services still install and run on Python 3.11
  (Raspberry Pi OS Bookworm) — they don't depend on `lerobot`.
- **OpenArm sim system deps:** `placo` (sim kinematics) dynamically links
  `liburdfdom`; install it from your distro's packages before running the sim.
- **Raspberry Pi install:** clone with `GIT_LFS_SKIP_SMUDGE=1` (or
  `git clone --filter=blob:none`) so Pi devices skip the OpenArm meshes, then
  `uv sync --package <service> --extra rpi`. See each package's README/Makefile.

## License

Apache-2.0. See [LICENSE](LICENSE).
