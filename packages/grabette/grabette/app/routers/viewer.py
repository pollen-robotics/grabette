"""3D URDF viewer endpoint — renders the grabette gripper with live joint angles."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from grabette.config import settings

router = APIRouter(tags=["viewer"])

_URDF_DIR = Path(__file__).resolve().parents[3] / "urdf"

# The page itself is a plain file so the getting-started page on GitHub Pages
# can publish the very same viewer (see .github/workflows/pages.yml).
VIEWER_HTML = (Path(__file__).resolve().parent.parent / "viewer.html").read_text()


@router.get("/viewer")
async def viewer():
    """Serve the 3D URDF viewer page.

    The URDF path is substituted at request time from settings.hand so
    a left-hand grabette renders grabette_left/ (mirror mesh) instead of
    the right one. The per-joint sign logic inside setJoint() is derived
    from URDF limits and works for either hand — the swap is cosmetic.

    It loads the light copy scripts/simplify_viewer_model.py builds (~4 MB
    instead of ~35 MB of CAD meshes), and the full model when that is absent.
    """
    model = f"grabette_{settings.hand}"
    if (_URDF_DIR / f"{model}_viewer" / "robot.urdf").is_file():
        model += "_viewer"
    urdf_path = f"/urdf/{model}/robot.urdf"
    return HTMLResponse(content=VIEWER_HTML.replace("__URDF_PATH__", urdf_path))
