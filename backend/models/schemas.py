from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum


class StatusEnum(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


# ─────────────────────────────────────────
# Upload
# ─────────────────────────────────────────

class UploadResponse(BaseModel):
    upload_id: str
    message: str
    files_received: List[str]


# ─────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    upload_id: str
    patient_id: Optional[str] = None   # optional override; defaults to upload_id


class AnalyzeResponse(BaseModel):
    upload_id: str
    patient_id: str
    status: StatusEnum
    message: str


# ─────────────────────────────────────────
# Segmentation result
# ─────────────────────────────────────────

class ClassDistribution(BaseModel):
    class_id: int
    label: str
    voxel_count: int
    percentage: float


class SegmentationResult(BaseModel):
    patient_id: str
    volume_shape: List[int]
    classes_detected: List[int]
    distribution: List[ClassDistribution]
    mask_path: str
    prob_path: str


# ─────────────────────────────────────────
# Volumetric result
# ─────────────────────────────────────────

class TumorRegionVolume(BaseModel):
    voxel_count: int
    volume_mm3: float
    volume_cm3: float
    percentage: float   # relative to whole tumor


class VolumetricResult(BaseModel):
    patient_id: str
    NET: TumorRegionVolume
    Edema: TumorRegionVolume
    ET: TumorRegionVolume
    whole_tumor_mm3: float
    whole_tumor_cm3: float
    tumor_core_mm3: float
    tumor_core_cm3: float
    nifti_path: str


# ─────────────────────────────────────────
# Full pipeline result
# ─────────────────────────────────────────

class PipelineResult(BaseModel):
    upload_id: str
    patient_id: str
    status: StatusEnum
    segmentation: Optional[SegmentationResult] = None
    volumetrics: Optional[VolumetricResult] = None
    error: Optional[str] = None


# ─────────────────────────────────────────
# Status / Health
# ─────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class JobStatus(BaseModel):
    upload_id: str
    status: StatusEnum
    progress: int = Field(ge=0, le=100, description="Completion percentage")
    current_step: Optional[str] = None
    error: Optional[str] = None