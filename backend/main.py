"""
CerebraScan AI — FastAPI Backend
──────────────────────────────────
Entry point.  Run with:

    uvicorn backend.main:app --reload --port 8000

or via Docker (see dockerfile / docker-compose.yaml).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes.upload  import router as upload_router
from backend.routes.analyze import router as analyze_router
from backend.routes.visualization import router as visualization_router
from backend.models.schemas import HealthResponse
from backend.services.segmentation import is_model_available
from backend.routes.mesh import router as mesh_router
from backend.routes.report import router as report_router

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from fastapi.responses import StreamingResponse
import asyncio
from backend.services.pipeline import LOG_STREAMS
import zipfile

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cerebrascan")


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═══ CerebraScan AI backend starting ═══")
    if is_model_available():
        logger.info("✓ Model checkpoint detected at checkpoints/seg_best.h5")
    else:
        logger.warning(
            "⚠  Model checkpoint NOT found. "
            "Run 'dvc repro train' before sending analysis requests."
        )
    yield
    logger.info("═══ CerebraScan AI backend shutting down ═══")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CerebraScan AI",
    description=(
        "End-to-end neuroimaging API for brain tumour segmentation, "
        "volumetric analysis, and NIfTI export using the BraTS U-Net pipeline."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (Frontend) ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(upload_router)
app.include_router(analyze_router)
app.include_router(visualization_router)
app.include_router(mesh_router)
app.include_router(report_router)

# ── Health & root ──────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(
        os.path.join(BASE_DIR, "templates", "index.html")
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"], summary="Health check")
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=is_model_available(),
    )


# ── Global exception handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check backend logs for details."},
    )

# ── Status endpoint (for live UI polling) ────────────────────────────────────
from backend.services.pipeline import get_job_status
from fastapi import APIRouter

status_router = APIRouter(prefix="/results")

@status_router.get("/{upload_id}/status")
def get_status(upload_id: str):
    status = get_job_status(upload_id)
    if not status:
        return {"status": "pending", "progress": 0}
    return status

app.include_router(status_router)

@app.get("/logs/{upload_id}")
async def stream_logs(upload_id: str):

    async def event_generator():
        last_index = 0
        while True:
            logs = LOG_STREAMS.get(upload_id, [])

            while last_index < len(logs):
                yield f"data: {logs[last_index]}\n\n"
                last_index += 1

            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(),
                             media_type="text/event-stream")

@app.get("/download/{upload_id}")
def download_visualizations(upload_id: str):

    folder = f"backend/storage/visualizations/{upload_id}"
    zip_path = f"{folder}.zip"

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in os.listdir(folder):
            zipf.write(os.path.join(folder, file),
                       arcname=file)

    return FileResponse(
        zip_path,
        filename=f"{upload_id}_visualizations.zip"
    )

@app.get("/mesh/{upload_id}/{region}")
async def get_mesh(upload_id: str, region: str):
    mesh_dir = Path("backend/storage/meshes")
    
    # Log what files exist
    all_files = list(mesh_dir.glob(f"{upload_id}_*"))
    print(f"Available mesh files: {[f.name for f in all_files]}")
    
    # Try PLY first (since that's what the errors show)
    ply_path = mesh_dir / f"{upload_id}_{region}.ply"
    if ply_path.exists():
        print(f"Serving PLY: {ply_path}")
        return FileResponse(ply_path, media_type="application/octet-stream")
    
    # Fallback to GLB
    glb_path = mesh_dir / f"{upload_id}_{region}.glb"
    if glb_path.exists():
        print(f"Serving GLB: {glb_path}")
        return FileResponse(glb_path, media_type="model/gltf-binary")
    
    raise HTTPException(status_code=404, detail=f"Mesh not found")