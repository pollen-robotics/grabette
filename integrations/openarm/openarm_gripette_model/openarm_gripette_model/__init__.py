"""OpenArm Gripette model — paths to URDF, MuJoCo XML, and assets."""

from pathlib import Path

# Root of the model data
MODEL_DIR = Path(__file__).resolve().parent

# Right arm model
OPENARM_RIGHT_DIR = MODEL_DIR / "openarm_right"
OPENARM_RIGHT_URDF = OPENARM_RIGHT_DIR / "robot.urdf"
OPENARM_RIGHT_XML = OPENARM_RIGHT_DIR / "robot.xml"
OPENARM_RIGHT_SCENE = OPENARM_RIGHT_DIR / "scene.xml"
OPENARM_RIGHT_ASSETS = OPENARM_RIGHT_DIR / "assets"
