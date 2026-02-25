"""
POST /analyze
──────────────
Triggers the full inference pipeline (preprocessing → segmentation → volumetrics)
for a previously uploaded set of MRI files.

The pipeline runs in a background thread.
Poll GET /results/{upload_id}/status to track progress.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from backend.models.schemas import AnalyzeRequest, AnalyzeResponse, StatusEnum
from backend.services.storage import list_uploaded_modalities, upload_dir
from backend.services.pipeline import launch_pipeline, get_job_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["Analysis"])

REQUIRED_MODALITIES = {"flair", "t1", "t1ce", "t2"}


@router.post(
    "/",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start analysis pipeline",
    description=(
        "Launch the preprocessing → segmentation → volumetric analysis pipeline "
        "for the given upload_id. The job runs asynchronously. "
        "Poll GET /results/{upload_id}/status for progress."
    ),
)
async def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
    upload_id  = body.upload_id
    patient_id = body.patient_id or upload_id

    # ── Validate upload exists ──────────────────────────────────────────────
    if not upload_dir(upload_id).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload '{upload_id}' not found. Upload MRI files first via POST /upload.",
        )

    # ── Check all modalities are present ────────────────────────────────────
    uploaded = set(list_uploaded_modalities(upload_id))
    missing  = REQUIRED_MODALITIES - uploaded

    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Missing modalities for upload '{upload_id}': {sorted(missing)}. "
                f"Uploaded so far: {sorted(uploaded)}. "
                "Re-upload with all four modalities (flair, t1, t1ce, t2)."
            ),
        )

    # ── Guard: don't start if already running ───────────────────────────────
    existing = get_job_status(upload_id)
    if existing and existing.get("status") == "processing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Pipeline for '{upload_id}' is already running. Poll /results/{upload_id}/status.",
        )

    # ── Launch ──────────────────────────────────────────────────────────────
    logger.info("[analyze] Launching pipeline — upload_id=%s  patient_id=%s",
                upload_id, patient_id)

    launch_pipeline(upload_id, patient_id)

    return AnalyzeResponse(
        upload_id=upload_id,
        patient_id=patient_id,
        status=StatusEnum.processing,
        message=(
            f"Pipeline started for upload '{upload_id}'. "
            f"Poll GET /results/{upload_id}/status for progress."
        ),
    )