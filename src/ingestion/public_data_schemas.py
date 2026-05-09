"""Section 1.9.3 Public Data Common Schemas.

Canonical schemas for reusing public datasets such as
KuaiRand/KuaiRec, LAMBDA/UltraLAMBDA, YouTube-8M on the same simulator.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Behavior Schema (Section 1.9.3.5)
#    KuaiRand / KuaiRec -> watch continuation, watch ratio, reaction signals
# ---------------------------------------------------------------------------


class BehaviorRecord(BaseModel):
    """Viewing behavior record.

    Normalizes and stores behavior data from KuaiRand / KuaiRec.
    """

    user_id: str
    item_id: str
    timestamp: datetime | None = None
    watch_time_ms: int = Field(default=0, ge=0, description="Watch time (milliseconds)")
    watch_ratio: float = Field(default=0.0, ge=0.0, description="Watch ratio (0.0~)")
    clicked: bool = False
    liked: bool = False
    left_session: bool = False
    # KuaiRand extension: 12 types of feedback signals
    feedback_signals: dict[str, float] = Field(
        default_factory=dict,
        description="Additional feedback signals (e.g. is_hate, is_not_interest, ...)",
    )
    source_dataset: str = Field(default="", description="Data source name (kuairand/kuairec)")


# ---------------------------------------------------------------------------
# 2. Ad Schema (Section 1.9.3.5)
#    LAMBDA / UltraLAMBDA -> ad recall, memorability score
# ---------------------------------------------------------------------------


class AdCreativeRecord(BaseModel):
    """Ad creative record.

    Normalizes and stores ad data from LAMBDA / UltraLAMBDA.
    """

    ad_id: str
    brand: str = ""
    duration_sec: float = Field(default=0.0, ge=0.0)
    memorability: float = Field(default=0.0, ge=0.0, le=1.0, description="memorability score")
    recall_score: float = Field(default=0.0, ge=0.0, le=1.0, description="recall score")
    scene_features: list[dict] = Field(
        default_factory=list,
        description="Scene-level features (e.g. [{scene_id, tags, ...}])",
    )
    ad_attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Additional attributes (category, tone, etc.)",
    )
    source_dataset: str = Field(default="", description="Data source name (lambda/ultralambda)")


# ---------------------------------------------------------------------------
# 3. Video Feature Schema (Section 1.9.3.5)
#    YouTube-8M -> categories, tags, audio-visual features
# ---------------------------------------------------------------------------


class VideoFeatureRecord(BaseModel):
    """Video feature record.

    Normalizes and stores video feature data from YouTube-8M.
    """

    video_id: str
    category: str = ""
    tags: list[str] = Field(default_factory=list)
    audio_visual_features: list[float] = Field(
        default_factory=list,
        description="audio-visual feature vector",
    )
    source: str = Field(default="", description="Data source name (youtube8m)")


# ---------------------------------------------------------------------------
# 4. Data Registry (Section 1.9.3.6)
# ---------------------------------------------------------------------------


class DataRegistryStatus(str, Enum):
    RAW = "raw"
    PREPROCESSED = "preprocessed"
    NORMALIZED = "normalized"


class DataRegistryEntry(BaseModel):
    """Data registry entry.

    Records management information for each acquired public dataset.
    """

    dataset_name: str = Field(description="Dataset name")
    source_url: str = Field(description="Source URL")
    acquired_date: datetime = Field(description="Acquisition date")
    license: str = Field(default="", description="License or terms of use")
    granularity: str = Field(default="", description="Granularity (user-item, ad-level, video-level)")
    primary_fields: list[str] = Field(default_factory=list, description="Primary fields")
    intended_use: str = Field(default="", description="Intended use")
    preprocessing_status: DataRegistryStatus = Field(
        default=DataRegistryStatus.RAW, description="Preprocessing status"
    )
    quality_notes: str = Field(default="", description="Quality notes")
    # Additional information for API usage
    api_quota_notes: str = Field(default="", description="Quota constraint notes")
    reacquisition_procedure: str = Field(default="", description="Reacquisition procedure")


class DataRegistry(BaseModel):
    """Data registry.

    Centrally manages metadata for all public datasets.
    """

    entries: list[DataRegistryEntry] = Field(default_factory=list)

