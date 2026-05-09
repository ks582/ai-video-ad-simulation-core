"""Ad pipeline common data models."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


# --- Ad metadata (FR-01) ---


class CTAType(str, Enum):
    WEBSITE_VISIT = "website_visit"
    APP_INSTALL = "app_install"
    PURCHASE = "purchase"
    SIGN_UP = "sign_up"
    LEARN_MORE = "learn_more"
    NONE = "none"
    # Detection-only types (emitted by detect_ctas_from_text, not selectable as
    # campaign metadata.cta_type). CTAEvent.cta_type is typed as ``str``, so
    # these enum values are stored as raw strings and downstream consumers
    # should match on the string value, not the enum identity.
    PHONE_CALL = "phone_call"
    SIGNUP = "signup"
    FREE_OFFER = "free_offer"


class CampaignObjective(str, Enum):
    AWARENESS = "awareness"
    CONSIDERATION = "consideration"
    CONVERSION = "conversion"


class AdMetadata(BaseModel):
    """Ad metadata."""

    brand: str
    category: str
    duration_sec: float = Field(gt=0)
    cta_type: CTAType
    target_audience: str = ""
    campaign_objective: CampaignObjective = CampaignObjective.AWARENESS


class AdInput(BaseModel):
    """Ad input (FR-01)."""

    input_id: str
    video_path: Path
    metadata: AdMetadata


# --- Video normalization output (FR-02) ---


class NormalizedVideo(BaseModel):
    """Normalized video output."""

    video_path: Path
    audio_path: Path
    frames_dir: Path
    fps: float
    total_frames: int
    duration_sec: float


# --- Shot segmentation (FR-03) ---


class Shot(BaseModel):
    """Shot — a contiguous range of an ad's timeline.

    Pre-2026-05-09 the model carried a ``representative_frame_paths``
    list that gated which frames the visual tagger and OCR could see.
    PR 2 of the Option 3 refactor moved ML inference to every 1 fps
    frame and PR 3 removed the field entirely; legacy ``result.json``
    files written before the bump still deserialize cleanly because
    Pydantic ignores unknown extras by default.
    """

    shot_id: int
    start_sec: float
    end_sec: float


# --- ASR (FR-04) ---


class ASRSegment(BaseModel):
    """Audio transcription segment."""

    start_sec: float
    end_sec: float
    text: str
    speaker_label: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# --- OCR (FR-05) ---


class OCRResult(BaseModel):
    """OCR result."""

    text: str
    bbox: list[list[int]] = Field(default_factory=list, description="[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]")
    time_sec: float
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    frame_path: Path | None = None


# --- Visual tags (FR-06) ---


class VisualTags(BaseModel):
    """Per-shot visual tags."""

    shot_id: int
    logo_visible: bool = False
    product_visible: bool = False
    cta_visible: bool = False
    emotion_tags: list[str] = Field(default_factory=list)
    scene_tags: list[str] = Field(default_factory=list)


class FrameVisualTags(BaseModel):
    """Per-frame visual tags (Option 3 — frame-level CLIP inference).

    Emitted by ``VisualTagger.tag_frames()`` — one entry per analyzed frame.
    The 1-indexed ``frame_idx`` matches the filename convention
    ``frame_NNNNN.png`` produced by ``VideoNormalizer``; ``time_sec`` is
    derived as ``(frame_idx - 1) / fps`` so that fps=1 yields a direct
    one-to-one correspondence with 1-second ``TimeBin`` entries (i.e.
    ``bin_id == frame_idx - 1``).

    This model is the new contract for downstream stages that need
    frame-precise visual signals — primarily the ``time_bins`` builder
    and the ``brand_exposure`` calculator. It coexists with ``VisualTags``
    (which remains as the per-shot aggregate). PR 2 of the Option 3
    refactor switches the pipeline to consume this model directly;
    PR 1 only adds the model + producer API.
    """

    frame_idx: int = Field(ge=1, description="1-indexed frame number")
    time_sec: float = Field(ge=0.0)
    logo_visible: bool = False
    product_visible: bool = False
    cta_visible: bool = False
    emotion_tags: list[str] = Field(default_factory=list)
    scene_tags: list[str] = Field(default_factory=list)


# --- Canonical Ad Schema (FR-07) ---


class TimeBin(BaseModel):
    """Time-bin (for agent observation)."""

    bin_id: int
    start_sec: float
    end_sec: float
    brand_logo_visible: bool = False
    product_visible: bool = False
    price_signal: bool = False
    cta_event: bool = False
    emotion_tone: str = "neutral"
    visual_complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    audio_energy: float = Field(default=0.5, ge=0.0, le=1.0)
    asr_text: str = ""
    ocr_text: str = ""


class BrandExposure(BaseModel):
    """Brand exposure information."""

    first_appearance_sec: float | None = None
    total_duration_sec: float = 0.0
    num_appearances: int = 0


class PriceSignalType(str, Enum):
    PRICE_DISPLAY = "price_display"
    DISCOUNT = "discount"
    FREE_TRIAL = "free_trial"
    LIMITED_OFFER = "limited_offer"


class PriceSignal(BaseModel):
    """Price appeal."""

    time_sec: float
    type: PriceSignalType
    text: str = ""


class CTAEvent(BaseModel):
    """CTA event."""

    time_sec: float
    cta_type: str
    cta_text: str = ""


class DominantTone(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    HUMOROUS = "humorous"
    DRAMATIC = "dramatic"
    INFORMATIVE = "informative"


class GlobalFeatures(BaseModel):
    """Overall ad features."""

    total_duration_sec: float
    num_shots: int
    dominant_tone: DominantTone = DominantTone.NEUTRAL
    has_music: bool = False
    has_narration: bool = False
    has_dialogue: bool = False


class CanonicalAdSchema(BaseModel):
    """Canonical Ad Schema (FR-07): Structured representation of an ad."""

    input_id: str
    metadata: AdMetadata
    global_features: GlobalFeatures
    shot_sequence: list[Shot]
    time_bins: list[TimeBin]
    brand_exposure: BrandExposure = Field(default_factory=BrandExposure)
    price_signals: list[PriceSignal] = Field(default_factory=list)
    cta_events: list[CTAEvent] = Field(default_factory=list)
    asr_segments: list[ASRSegment] = Field(default_factory=list)
    ocr_results: list[OCRResult] = Field(default_factory=list)
    visual_tags: list[VisualTags] = Field(default_factory=list)
    # Frame-level visual tag results (Option 3). Populated by the
    # PR 2 pipeline switch; legacy ``result.json`` files written before
    # 2026-05-09 lack this key and deserialize with the empty default.
    # ``visual_tags`` above is now synthesized post-hoc from this list
    # (per-shot union over the frames falling inside the shot's time
    # range) so the existing field stays populated for backward-compatible
    # consumers that read it directly.
    frame_visual_tags: list[FrameVisualTags] = Field(default_factory=list)
