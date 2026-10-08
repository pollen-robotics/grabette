"""3D URDF viewer endpoint — renders the grabette gripper with live joint angles."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from grabette.config import settings

router = APIRouter(tags=["viewer"])


def _viewer_page() -> str:
    """The viewer page, with the model this grabette loads.

    The page itself is a plain file so the getting-started page on GitHub Pages
    can publish the very same viewer (see .github/workflows/pages.yml).

    The URDF path comes from settings.hand so a left-hand grabette renders
    grabette_left/ (mirror mesh) instead of the right one. The per-joint sign
    logic inside setJoint() is derived from URDF limits and works for either
    hand — the swap is cosmetic. It is the light copy
    scripts/simplify_viewer_model.py builds (~4 MB instead of ~35 MB of CAD
    meshes), and the full model when that is absent.
    """
    here = Path(__file__).resolve()
    model = f"grabette_{settings.hand}"
    if (here.parents[3] / "urdf" / f"{model}_viewer" / "robot.urdf").is_file():
        model += "_viewer"
    html = (here.parent.parent / "viewer.html").read_text()
    return html.replace("__URDF_PATH__", f"/urdf/{model}/robot.urdf")


# Neither the hand nor the files change while the daemon runs.
_VIEWER_PAGE = _viewer_page()


@router.get("/viewer")
async def viewer():
    """Serve the 3D URDF viewer page."""
    return HTMLResponse(content=_VIEWER_PAGE)
