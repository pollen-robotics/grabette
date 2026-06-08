#!/usr/bin/env python3
"""Post-process onshape-to-robot MuJoCo output for the OpenArm Gripette model.

Run this after every `onshape-to-robot-mujoco openarm_right` generation.

Patches to robot.xml:
  1. Remove meshdir="assets" from <compiler> and prefix mesh filenames with "assets/"
     → makes robot.xml includable from scene files in other directories
  2. Set contype="2" conaffinity="1" on the collision geom class
     → prevents self-collision between adjacent robot links
  3. Increase armature (0.005 → 0.1) and actuator gains (kp 50 → 500)
     → stiffer position tracking in simulation
  4. Add the Gripette camera element next to the camera site
     → MuJoCo camera for offscreen rendering (with 180° pitch correction, fovy=130)

Patches to scene.xml:
  5. Add offwidth="1296" offheight="972" to <global> so offscreen rendering works

Usage:
    cd openarm_gripette_model
    uv run onshape-to-robot-mujoco openarm_right
    python postprocess_mujoco.py
"""

import re
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "openarm_gripette_model" / "openarm_right"
ROBOT_XML = MODEL_DIR / "robot.xml"
SCENE_XML = MODEL_DIR / "scene.xml"

CAMERA_ELEMENT = (
    '                    <!-- Gripette camera (180° pitch correction from site frame to MuJoCo camera convention) -->\n'
    '                    <!-- fovy=130° for wide pinhole render, fisheye distortion applied in post -->\n'
    '                    <camera name="gripette_cam" pos="0.0687999 0.0662004 0.0260816"'
    ' quat="0.500002 0.499998 0.499998 -0.500002" fovy="130"/>'
)


def patch_robot_xml():
    text = ROBOT_XML.read_text()
    original = text

    # 1. Remove meshdir from compiler, prefix mesh files
    text = re.sub(r'(<compiler\b[^>]*)\s+meshdir="assets"([^>]*>)', r'\1\2', text)
    text = re.sub(r'<mesh file="(?!assets/)([^"]+)"', r'<mesh file="assets/\1"', text)

    # 2. Collision class filtering
    text = re.sub(
        r'(<default class="collision">\s*<geom group="3")(\s*/>)',
        r'\1 contype="2" conaffinity="1"\2',
        text,
    )

    # 3. Stiffer actuators
    text = re.sub(
        r'<joint frictionloss="0\.1" armature="0\.005"/>',
        '<joint frictionloss="0.1" armature="0.1"/>',
        text,
    )
    text = re.sub(r'<position kp="50"', '<position kp="500"', text)

    # 4. Gripette camera
    if "gripette_cam" not in text:
        pat = r'(<!-- Frame camera -->\n\s*<site group="3" name="camera"[^/]*/>\n)'
        m = re.search(pat, text)
        if m:
            text = text[:m.end()] + CAMERA_ELEMENT + "\n" + text[m.end():]
        else:
            print("WARNING: camera site not found — skipping gripette_cam insertion")

    if text != original:
        ROBOT_XML.write_text(text)
        print(f"Patched {ROBOT_XML.name}")
    else:
        print(f"No changes in {ROBOT_XML.name}")


def patch_scene_xml():
    if not SCENE_XML.exists():
        print(f"{SCENE_XML} not found — skipping")
        return
    text = SCENE_XML.read_text()
    original = text

    # 5. Add offwidth/offheight to <global> tag (for offscreen rendering)
    if "offwidth" not in text:
        text = re.sub(
            r'(<global\b[^/]*?)(/?>)',
            r'\1 offwidth="1296" offheight="972"\2',
            text,
            count=1,
        )

    if text != original:
        SCENE_XML.write_text(text)
        print(f"Patched {SCENE_XML.name}")
    else:
        print(f"No changes in {SCENE_XML.name}")


def main():
    patch_robot_xml()
    patch_scene_xml()


if __name__ == "__main__":
    main()
