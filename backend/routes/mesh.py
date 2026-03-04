from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()

MESH_DIR = Path("backend/storage/meshes")


@router.get("/mesh/{case_id}/{region}")
def get_mesh(case_id: str, region: str):

    mesh_path = MESH_DIR / f"{case_id}_{region}.ply"

    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="Mesh not found")

    return FileResponse(mesh_path, media_type="application/octet-stream")