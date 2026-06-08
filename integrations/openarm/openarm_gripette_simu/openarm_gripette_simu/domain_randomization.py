"""Per-episode visual domain randomization for MuJoCo scenes.

Why this module exists
----------------------
A scripted demonstration collector generates trajectories that all look
nearly identical to the camera (same lighting, same table, same camera pose).
A diffusion policy trained on such data can learn the *motion template* from
proprioception/action history alone and barely consult the camera. At
deployment time the slightest visual perturbation — different lighting in
the lab, a slightly nudged camera mount, a different table colour — breaks
the policy.

Domain randomization at episode boundaries forces the visual encoder to be
*invariant to nuisance variation* without changing the task itself: the
cube, the goal, and the success criterion all stay the same. Only
irrelevant visual factors (light, table colour, camera pose) move around.
This is the cheap version of UMI-style randomization, scoped to nuisance
variables only — distractors and multi-object selection are out of scope.

Robot-agnostic. Works on any MuJoCo `model` that exposes a named light, a
named camera and (optionally) named materials. The DR functions mutate the
model in place, so calling them once per `Simulation()` instance is enough
— each Simulation reload rebuilds the model from XML, which restores
defaults.

Example
-------
    from openarm_gripette_simu import Simulation
    from openarm_gripette_simu.domain_randomization import DRConfig, randomize_scene

    cfg = DRConfig(
        headlight_intensity_range=(0.7, 1.3),
        light_dir_jitter_rad=np.deg2rad(15.0),
        material_rgb_jitter=0.10,
        material_names=("wood",),
        camera_pos_jitter_m=0.01,
        camera_names=("gripette_cam",),
    )
    rng = np.random.default_rng(0)

    sim = Simulation(scene_xml=...)
    info = randomize_scene(sim.model, rng, cfg)  # returns the actual sampled values
    # ... run episode normally ...
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class DRConfig:
    """Per-episode visual nuisance randomization parameters.

    All ranges are sampled uniformly per episode. Set any range to its
    no-op value (1.0 multiplier, 0.0 jitter, empty tuple of names) to skip
    that channel.
    """
    # Headlight diffuse multiplier (camera-attached ambient/diffuse light).
    # Range = (lo, hi); each episode samples one scalar applied to all 3 channels.
    headlight_intensity_range: tuple[float, float] = (1.0, 1.0)

    # Directional light direction jitter. Each episode samples a unit-vector
    # offset in a cone around the original direction with this max half-angle.
    light_dir_jitter_rad: float = 0.0
    light_names: tuple[str, ...] = ()  # named lights to jitter; empty = skip

    # Per-channel additive RGB jitter on listed materials.
    # Each material's rgb is offset by U(-jitter, +jitter) per channel.
    # Alpha is left untouched. Values are clipped to [0, 1].
    material_rgb_jitter: float = 0.0
    material_names: tuple[str, ...] = ()  # empty = skip

    # Per-axis Gaussian-clipped uniform jitter on listed cameras' parent-body
    # offsets. Magnitude in metres.
    camera_pos_jitter_m: float = 0.0
    camera_names: tuple[str, ...] = ()  # empty = skip


# --------------------------------------------------------------------------- #
# Per-channel helpers
# --------------------------------------------------------------------------- #


def _randomize_headlight(model, rng: np.random.Generator, lo: float, hi: float) -> float:
    """Multiply headlight diffuse by a per-episode scalar; return the multiplier."""
    if (lo, hi) == (1.0, 1.0):
        return 1.0
    mult = float(rng.uniform(lo, hi))
    # `model.vis.headlight.diffuse` is a 3-vector. We don't know its prior
    # value if this function is called twice on the same model — but each
    # Simulation reload reads the XML afresh, so this is safe within the
    # "once per episode" contract.
    model.vis.headlight.diffuse[:] = np.clip(model.vis.headlight.diffuse * mult, 0.0, 1.0)
    model.vis.headlight.ambient[:] = np.clip(model.vis.headlight.ambient * mult, 0.0, 1.0)
    return mult


def _sample_unit_vector_in_cone(axis: np.ndarray, half_angle_rad: float,
                                rng: np.random.Generator) -> np.ndarray:
    """Sample a unit vector uniformly in a cone of given half-angle around `axis`.

    The cone-of-directions sampling: pick a polar angle in [0, half_angle]
    uniformly in cos(theta), pick an azimuth in [0, 2π], then rotate that
    sample so the cone axis lines up with `axis`.
    """
    if half_angle_rad <= 0.0:
        return axis / np.linalg.norm(axis)
    cos_max = np.cos(half_angle_rad)
    cos_t = rng.uniform(cos_max, 1.0)
    sin_t = np.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    phi = rng.uniform(0.0, 2.0 * np.pi)
    # Local-frame sample (cone axis = +Z)
    local = np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])
    # Rotate local so its +Z aligns with `axis`. Use Rodrigues / cross-product.
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    z_hat = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_hat, axis)
    s = np.linalg.norm(v)
    c = float(np.dot(z_hat, axis))
    if s < 1e-9:
        # axis is parallel or antiparallel to +Z
        return local if c > 0 else -local
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s * s))
    return R @ local


def _randomize_lights(model, rng: np.random.Generator, half_angle_rad: float,
                      light_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Jitter named lights' direction within a cone. Returns the new directions."""
    out: dict[str, np.ndarray] = {}
    if half_angle_rad <= 0.0 or not light_names:
        return out
    for name in light_names:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, name)
        if idx < 0:
            raise ValueError(f"Light '{name}' not found in model")
        old_dir = model.light_dir[idx].copy()
        new_dir = _sample_unit_vector_in_cone(old_dir, half_angle_rad, rng)
        model.light_dir[idx] = new_dir
        out[name] = new_dir
    return out


def _randomize_materials(model, rng: np.random.Generator, jitter: float,
                         material_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Add per-channel uniform RGB offset to named materials. Alpha untouched."""
    out: dict[str, np.ndarray] = {}
    if jitter <= 0.0 or not material_names:
        return out
    for name in material_names:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if idx < 0:
            raise ValueError(f"Material '{name}' not found in model")
        offset = rng.uniform(-jitter, +jitter, size=3)
        new_rgb = np.clip(model.mat_rgba[idx, :3] + offset, 0.0, 1.0)
        model.mat_rgba[idx, :3] = new_rgb
        out[name] = new_rgb.copy()
    return out


def _randomize_cameras(model, rng: np.random.Generator, jitter_m: float,
                       camera_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Add per-axis uniform position offset to named cameras' parent-body anchor."""
    out: dict[str, np.ndarray] = {}
    if jitter_m <= 0.0 or not camera_names:
        return out
    for name in camera_names:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if idx < 0:
            raise ValueError(f"Camera '{name}' not found in model")
        offset = rng.uniform(-jitter_m, +jitter_m, size=3)
        new_pos = model.cam_pos[idx] + offset
        model.cam_pos[idx] = new_pos
        out[name] = new_pos.copy()
    return out


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def randomize_scene(model, rng: np.random.Generator, cfg: DRConfig) -> dict:
    """Apply all configured DR channels to `model` in place. Returns a dict
    of the actually-sampled values for logging / dataset metadata.

    Idempotency note: if you call this twice on the same model the jitter
    compounds. The expected pattern is to instantiate a fresh `Simulation`
    per episode (which reloads the XML, restoring defaults) and call this
    function once.
    """
    info: dict = {}
    info["headlight_mult"] = _randomize_headlight(
        model, rng, *cfg.headlight_intensity_range,
    )
    info["light_dirs"] = _randomize_lights(
        model, rng, cfg.light_dir_jitter_rad, cfg.light_names,
    )
    info["material_rgbs"] = _randomize_materials(
        model, rng, cfg.material_rgb_jitter, cfg.material_names,
    )
    info["camera_pos"] = _randomize_cameras(
        model, rng, cfg.camera_pos_jitter_m, cfg.camera_names,
    )
    return info
