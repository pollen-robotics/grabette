from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/grpc", tags=["grpc"])


@router.get("/status")
def grpc_status():
    from grabette.app.main import get_grpc_server_instance

    srv = get_grpc_server_instance()
    if srv is None:
        return {"enabled": False, "connected": False}
    return srv.get_status()
