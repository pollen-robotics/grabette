"""The dashboard: one static page, no framework.

Gradio (grabette's dashboard) is too heavy for the 512 MB Pi Zero 2W, so this is
a single HTML file with plain JavaScript polling the JSON API, like gripette's
status page. Pages are tabs inside it (#overview, #episodes, #network).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["ui"])

_DASHBOARD = Path(__file__).resolve().parent.parent.parent / "static" / "dashboard.html"


@router.get("/", include_in_schema=False)
def dashboard():
    # no-cache: the page is tiny, and a stale copy after an update would talk
    # to endpoints that changed under it.
    return FileResponse(_DASHBOARD, media_type="text/html",
                        headers={"Cache-Control": "no-cache"})
