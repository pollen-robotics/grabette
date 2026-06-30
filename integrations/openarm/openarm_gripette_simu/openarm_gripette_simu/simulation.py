"""MuJoCo simulation wrapper for the OpenArm right arm + Gripette.

Handles physics stepping, joint control, state readback, and camera rendering.
Joint behavior (gains, damping, friction) is tuned in the MuJoCo XML model.
"""

from pathlib import Path
import re
import tempfile
import cv2
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
    """Load a MuJoCo model, resolving the robot include's mesh paths.

    MuJoCo ignores `meshdir` from <include>d files and resolves their bare
    mesh filenames relative to the included file's own directory — missing
    the `assets/` subdir the export uses (robot.xml has meshdir="assets").
    So we inline the robot file in place of the <include>, with its meshdir
    rewritten to an absolute path. Derived from the robot file itself, this
    works for both the `assets/` layout and the older flat layout.
    """
    xml = scene_xml.read_text()
    inc = re.search(r'<include\s+file="([^"]*)"\s*/>', xml)
    if inc is None:
        return mujoco.MjModel.from_xml_path(str(scene_xml))

    robot_path = (scene_xml.parent / inc.group(1)).resolve()
    robot_xml = robot_path.read_text()

    # Make the robot's meshdir absolute (it is relative to the robot file dir).
    md = re.search(r'meshdir="([^"]*)"', robot_xml)
    abs_meshdir = (robot_path.parent / (md.group(1) if md else ".")).resolve()
    if md:
        robot_xml = robot_xml.replace(md.group(0), f'meshdir="{abs_meshdir}"')

    # Inline the robot file's body where the <include> was.
    inner = re.search(r'<mujoco[^>]*>(.*)</mujoco>', robot_xml, re.S).group(1)
    xml = xml.replace(inc.group(0), inner)

    # The export gives the gripper a "camera" SITE (the calibrated frame) but no
    # MuJoCo <camera>, so render_camera() would fall back to the free camera.
    # Inject a gripette_cam at that site unless the scene already defines one
    # (the free-floating scene generates its own). MuJoCo cameras look along -z
    # while the site's optical axis is +z, so rotate 180° about x — matching the
    # free-floating scene generator's convention.
    if 'name="gripette_cam"' not in xml:
        site_m = re.search(r'<site\b[^>]*\bname="camera"[^>]*/>', xml)
        if site_m:
            site_el = site_m.group(0)
            pos_m = re.search(r'pos="([^"]*)"', site_el)
            quat_m = re.search(r'quat="([^"]*)"', site_el)
            pos = pos_m.group(1) if pos_m else "0 0 0"
            sq = (np.array([float(v) for v in quat_m.group(1).split()])
                  if quat_m else np.array([1.0, 0.0, 0.0, 0.0]))
            cq = np.zeros(4)
            mujoco.mju_mulQuat(cq, sq, np.array([0.0, 1.0, 0.0, 0.0]))
            cam_el = (f'<camera name="gripette_cam" pos="{pos}" '
                      f'quat="{cq[0]:.7g} {cq[1]:.7g} {cq[2]:.7g} {cq[3]:.7g}" '
                      f'fovy="130"/>')
            xml = xml.replace(site_el, site_el + cam_el)

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

        # Render option for the camera: hide ALL site groups so reference frames
        # (thumb_tip / finger_tip / gripper_center) never appear in the recorded
        # images. Sites are frames-only, never part of the policy observation.
        self._cam_opt = mujoco.MjvOption()
        self._cam_opt.sitegroup[:] = 0

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

    def get_actuator_ctrl(self, joint_names: list[str] | None = None) -> np.ndarray:
        """Read the current actuator position COMMANDS (data.ctrl, not qpos)."""
        if joint_names is None:
            joint_names = ACTUATOR_NAMES
        return np.array([self.data.ctrl[self._actuator_ids[name]] for name in joint_names])

    def render_camera(self, out_size: tuple[int, int] | None = None) -> np.ndarray:
        """Render an image from the Gripette camera with fisheye distortion.

        Args:
            out_size: optional (width, height) to downscale the distorted image
                to (INTER_AREA). The fisheye distortion is always applied at the
                calibration resolution (1296x972) first; this only shrinks the
                result — used to match the real Grabette stream (960x720). The
                resolutions are all 4:3, so the uniform downscale keeps the KB8
                calibration valid. Default None returns native 972x1296.

        Returns:
            RGB uint8 array of shape (out_h, out_w, 3), or (972, 1296, 3).
        """
        self._renderer.update_scene(self.data, camera=self._cam_id,
                                    scene_option=self._cam_opt)
        img = self._fisheye.distort(self._renderer.render())
        if out_size is not None and (img.shape[1], img.shape[0]) != out_size:
            img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
        return img

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
