"""
GET /results/{upload_id}/status   — poll job progress
GET /results/{upload_id}          — fetch full pipeline result
GET /results/{upload_id}/nifti    — download exported NIfTI file
"""

import logging
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from backend.models.schemas import (
    JobStatus,
    PipelineResult,
    SegmentationResult,
    VolumetricResult,
    TumorRegionVolume,
    ClassDistribution,
    StatusEnum,
)
from backend.services.pipeline import get_job_status
from backend.services.storage  import nifti_output_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/results", tags=["Results"])


def _require_job(upload_id: str) -> dict:
    job = get_job_status(upload_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No pipeline job found for upload_id '{upload_id}'. "
                "Start one via POST /analyze."
            ),
        )
    return job


# ── Status endpoint ────────────────────────────────────────────────────────────

@router.get(
    "/{upload_id}/status",
    response_model=JobStatus,
    summary="Poll pipeline progress",
    description="Returns current status and progress percentage (0-100) for a running pipeline.",
)
async def get_status(upload_id: str) -> JobStatus:
    job = _require_job(upload_id)
    return JobStatus(
        upload_id=upload_id,
        status=StatusEnum(job.get("status", "pending")),
        progress=job.get("progress", 0),
        current_step=job.get("current_step"),
        error=job.get("error"),
    )


# ── Full result endpoint ───────────────────────────────────────────────────────

@router.get(
    "/{upload_id}",
    response_model=PipelineResult,
    summary="Fetch full pipeline result",
    description=(
        "Returns segmentation and volumetric results once the pipeline has completed. "
        "Returns 202 if the pipeline is still running."
    ),
)
async def get_result(upload_id: str) -> PipelineResult:
    job       = _require_job(upload_id)
    job_status = StatusEnum(job.get("status", "pending"))
    patient_id = job.get("patient_id", upload_id)

    if job_status == StatusEnum.processing or job_status == StatusEnum.pending:
        # Still running – return partial state with 202
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "upload_id":  upload_id,
                "patient_id": patient_id,
                "status":     job_status.value,
                "message":    f"Pipeline {job_status.value} — {job.get('current_step', '')}",
                "progress":   job.get("progress", 0),
            },
        )

    if job_status == StatusEnum.failed:
        return PipelineResult(
            upload_id=upload_id,
            patient_id=patient_id,
            status=job_status,
            error=job.get("error"),
        )

    # ── Completed — build structured response ──────────────────────────────
    seg_raw = job.get("segmentation")
    vol_raw = job.get("volumetrics")

    seg_result = None
    if seg_raw:
        seg_result = SegmentationResult(
            patient_id=seg_raw["patient_id"],
            volume_shape=seg_raw["volume_shape"],
            classes_detected=seg_raw["classes_detected"],
            distribution=[
                ClassDistribution(**d) for d in seg_raw["distribution"]
            ],
            mask_path=seg_raw["mask_path"],
            prob_path=seg_raw["prob_path"],
        )

    vol_result = None
    if vol_raw:
        vol_result = VolumetricResult(
            patient_id=vol_raw["patient_id"],
            NET=TumorRegionVolume(**vol_raw["NET"]),
            Edema=TumorRegionVolume(**vol_raw["Edema"]),
            ET=TumorRegionVolume(**vol_raw["ET"]),
            whole_tumor_mm3=vol_raw["whole_tumor_mm3"],
            whole_tumor_cm3=vol_raw["whole_tumor_cm3"],
            tumor_core_mm3=vol_raw["tumor_core_mm3"],
            tumor_core_cm3=vol_raw["tumor_core_cm3"],
            nifti_path=vol_raw["nifti_path"],
        )

    return PipelineResult(
        upload_id=upload_id,
        patient_id=patient_id,
        status=job_status,
        segmentation=seg_result,
        volumetrics=vol_result,
    )


# ── NIfTI download endpoint ────────────────────────────────────────────────────

@router.get(
    "/{upload_id}/nifti",
    summary="Download segmentation NIfTI",
    description="Download the exported NIfTI (.nii.gz) segmentation file.",
)
async def download_nifti(upload_id: str):
    job = _require_job(upload_id)

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="NIfTI is only available once the pipeline has completed.",
        )

    patient_id = job.get("patient_id", upload_id)
    nifti_path = nifti_output_path(upload_id, patient_id)

    if not nifti_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="NIfTI file not found. The pipeline may have failed during export.",
        )

    return FileResponse(
        path=str(nifti_path),
        media_type="application/gzip",
        filename=f"{patient_id}_segmentation.nii.gz",
    )s