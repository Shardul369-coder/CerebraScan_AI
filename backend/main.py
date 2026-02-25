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

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(upload_router)
app.include_router(analyze_router)
app.include_router(visualization_router)

# ── Health & root ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"], summary="Root")
async def root():
    return {"message": "CerebraScan AI backend is running. Visit /docs for the API."}


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