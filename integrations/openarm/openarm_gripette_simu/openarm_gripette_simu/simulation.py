"""MuJoCo simulation wrapper for the OpenArm right arm + Gripette.

Handles physics stepping, joint control, state readback, and camera rendering.
Joint behavior (gains, damping, friction) is tuned in the MuJoCo XML model.
"""

from pathlib import Path
import re
import tempfile
import numpy as np
import mujoco
import mujoco.viewer
from openarm_gripette_model import OPENARM_RIGHT_DIR, OPENARM_RIGHT_SCENE
from .camera import FisheyeCamera
from .kinematics import ARM_JOINT_NAMES

# All actuated joints in MuJoCo ordering
ACTUATOR_NAMES = [*ARM_JOINT_NAMES, "proximal", "distal"]

# Gripette camera name (as defined in robot.xml)
GRIPETTE_CAM = "gripette_cam"


def _load_model(scene_xml: Path) -> mujoco.MjModel:
    """Load a MuJoCo model, injecting meshdir when the scene is outside the model dir.

    MuJoCo resolves mesh paths relative to the main XML file. When a scene
    in a different directory includes robot.xml, the mesh paths break.
    This injects an absolute meshdir so meshes are always found.
    """
    xml = scene_xml.read_text()

    # Only inject meshdir if not already set in the scene
    if 'meshdir=' not in xml.split('<include')[0]:
        meshdir_tag = f'<compiler meshdir="{OPENARM_RIGHT_DIR}"/>'
        xml = re.sub(r'(<include\s)', meshdir_tag + r'\n    \1', xml, count=1)
        # Write temp file next to the scene so relative includes still resolve
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.xml', dir=scene_xml.parent, delete=False
        ) as f:
            f.write(xml)
            tmp_path = Path(f.name)
        try:
            return mujoco.MjModel.from_xml_path(str(tmp_path))
        finally:
            tmp_path.unlink()

    return mujoco.MjModel.from_xml_path(str(scene_xml))


class Simulation:
    """MuJoCo simulation of the OpenArm + Gripette."""

    def __init__(self, scene_xml: str | Path | None = None):
        scene_xml = Path(scene_xml).resolve() if scene_xml else OPENARM_RIGHT_SCENE
        self.model = _load_model(scene_xml)
        self.data = mujoco.MjData(self.model)

        # Actuator name -> index mapping
        self._actuator_ids = {
            self.model.actuator(i).name: i for i in range(self.model.nu)
        }

        # Fisheye camera model (precomputes remap tables)
        self._fisheye = FisheyeCamera()
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, GRIPETTE_CAM)

        # Create the offscreen renderer eagerly so its GL context is
        # initialized before the viewer (avoids GLX threading conflicts)
        self._renderer = mujoco.Renderer(
            self.model,
            height=self._fisheye.pinhole_height,
            width=self._fisheye.pinhole_width,
        )

    def reset_joints(self, positions: np.ndarray, joint_names: list[str] | None = None):
        """Teleport joints to the given positions (no physics stepping).

        Sets qpos directly, updates actuator targets to match, and
        recomputes all derived quantities. No collision with the environment.
        """
        if joint_names is None:
            joint_names = ACTUATOR_NAMES
        for name, pos in zip(joint_names, positions):
            self.data.joint(name).qpos[0] = pos
        # Also set the mimic joint if r_wrist_roll is being set
        if "r_wrist_roll" in joint_names:
            idx = joint_names.index("r_wrist_roll")
            self.data.joint("r_wrist_roll_mimic").qpos[0] = -positions[idx]
        # Zero velocities
        self.data.qvel[:] = 0
        # Set actuator targets to match so the arm holds position
        self.set_joint_commands(positions, joint_names)
        # Recompute all derived quantities (positions, contacts, etc.)
        mujoco.mj_forward(self.model, self.data)

    def reset_arm(self, positions: np.ndarray):
        """Teleport the 7 arm joints to the given positions."""
        self.reset_joints(positions, ARM_JOINT_NAMES)

    def step(self):
        """Advance the simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

    def set_joint_commands(self, commands: np.ndarray, joint_names: list[str] | None = None):
        """Set position commands for the actuators."""
        if joint_names is None:
            joint_names = ACTUATOR_NAMES
        for name, cmd in zip(joint_names, commands):
            self.data.ctrl[self._actuator_ids[name]] = cmd

    def set_arm_commands(self, commands: np.ndarray):
        """Set position commands for the 7 arm joints only."""
        self.set_joint_commands(commands, ARM_JOINT_NAMES)

    def get_joint_positions(self, joint_names: list[str] | None = None) -> np.ndarray:
        """Read current joint positions from the simulation."""
        if joint_names is None:
            joint_names = ACTUATOR_NAMES
        return np.array([self.data.joint(name).qpos[0] for name in joint_names])

    def get_arm_positions(self) -> np.ndarray:
        """Read current arm joint positions (7 values)."""
        return self.get_joint_positions(ARM_JOINT_NAMES)

    def render_camera(self) -> np.ndarray:
        """Render an image from the Gripette camera with fisheye distortion.

        Returns:
            RGB uint8 array of shape (972, 1296, 3).
        """
        self._renderer.update_scene(self.data, camera=self._cam_id)
        return self._fisheye.distort(self._renderer.render())

    def launch_viewer(self):
        """Launch the interactive MuJoCo viewer (blocking)."""
        mujoco.viewer.launch(self.model, self.data)

    def launch_passive_viewer(self, show_left_ui: bool = False, show_right_ui: bool = False):
        """Launch a passive MuJoCo viewer (non-blocking).

        Side panels are hidden by default — most scripted demos don't need
        them and the extra screen real estate is more useful. Pass
        `show_left_ui=True` / `show_right_ui=True` to restore them.

        Returns the viewer handle. Call viewer.sync() after each step.
        """
        return mujoco.viewer.launch_passive(
            self.model, self.data,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        )
