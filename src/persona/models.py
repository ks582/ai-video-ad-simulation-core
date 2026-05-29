"""T-201: Persona / Agent data models (FR-08, FR-15).

3-stage population pipeline models:
  P(x)         → SegmentAttrs + DigitalProfile (public-stats demographics)
  P(z_pre|x)   → ProcessingProfile + BrandPrior + LatentPreExposureState (GenAI baseline prior)
  Simulation   → MemoryState (dynamic state base)

Twin-world support (FR-09B): BaselineDraw + TwinPairing.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared scalar type
# ---------------------------------------------------------------------------


class StatusEnum(str, Enum):
    OBSERVED = "observed"
    PROXY = "proxy"
    ASSUMED = "assumed"
    UNKNOWN = "unknown"
    DERIVED = "derived"


class DistributedScalar(BaseModel):
    """A scalar estimate with uncertainty bounds and provenance annotation.

    Used for any field inferred from public data or GenAI rather than directly
    observed.  Satisfies FR-09A requirement that baseline prior fields return
    distributions, not point estimates.
    """

    mean: float
    std: float | None = None
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    status: StatusEnum
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: str | None = None

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# P(x) — demographic / device blocks
# ---------------------------------------------------------------------------


class AgeGroup(str, Enum):
    A18_24 = "18-24"
    A25_34 = "25-34"
    A35_44 = "35-44"
    A45_54 = "45-54"
    A55_64 = "55-64"
    A65_PLUS = "65+"


class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"
    UNSPECIFIED = "unspecified"


class IncomeBracket(str, Enum):
    UNDER_25K = "<25k"
    K25_50 = "25k-50k"
    K50_75 = "50k-75k"
    K75_100 = "75k-100k"
    K100_150 = "100k-150k"
    OVER_150K = "150k+"


class EducationLevel(str, Enum):
    LESS_THAN_HS = "less_than_hs"
    HS_DIPLOMA = "hs_diploma"
    SOME_COLLEGE = "some_college"
    BACHELORS = "bachelors"
    GRADUATE = "graduate"


class RegionEnum(str, Enum):
    NORTHEAST = "northeast"
    MIDWEST = "midwest"
    SOUTH = "south"
    WEST = "west"


class EmploymentStatus(str, Enum):
    EMPLOYED_FULL = "employed_full_time"
    EMPLOYED_PART = "employed_part_time"
    UNEMPLOYED = "unemployed"
    NOT_IN_LF = "not_in_labor_force"
    STUDENT = "student"


class HouseholdType(str, Enum):
    SINGLE = "single"
    COUPLE_NO_CHILDREN = "couple_no_children"
    FAMILY_WITH_CHILDREN = "family_with_children"
    SINGLE_PARENT = "single_parent"
    MULTIGENERATIONAL = "multigenerational"


class SegmentAttrs(BaseModel):
    """Demographic attributes from public statistics P(x).

    Sourced from the American Community Survey (ACS) and the NTIA Internet
    Use Survey. Defines the poststratification cell for IPF weighting.
    """

    age_group: AgeGroup
    gender: GenderEnum
    income_bracket: IncomeBracket
    education_level: EducationLevel
    region: RegionEnum
    employment_status: EmploymentStatus
    household_type: HouseholdType | None = None
    poststratification_cell_id: str | None = None
    cell_weight: float | None = Field(default=None, ge=0.0)


class DevicePreference(str, Enum):
    MOBILE = "mobile"
    DESKTOP = "desktop"
    TABLET = "tablet"
    CONNECTED_TV = "connected_tv"
    MIXED = "mixed"


class InternetAccessType(str, Enum):
    BROADBAND = "broadband"
    MOBILE_ONLY = "mobile_only"
    LIMITED = "limited"
    NONE = "none"


class AudioPreference(str, Enum):
    SOUND_ON = "sound_on"
    SOUND_OFF = "sound_off"
    HEADPHONES = "headphones"
    VARIES = "varies"


class DigitalProfile(BaseModel):
    """Device and internet usage attributes (NTIA, ACS)."""

    primary_device: DevicePreference = DevicePreference.MOBILE
    internet_access_type: InternetAccessType = InternetAccessType.BROADBAND
    daily_online_video_minutes: DistributedScalar | None = None
    audio_setting_preference: AudioPreference = AudioPreference.SOUND_ON


# ---------------------------------------------------------------------------
# P(x) — category / behavioral baseline blocks
# ---------------------------------------------------------------------------


class CategoryProfile(BaseModel):
    """Content interest distribution and category spending profile."""

    interest_vector: list[float] = Field(min_length=1)
    interest_categories: list[str] = Field(default_factory=list)
    category_spend_indices: dict[str, float] = Field(default_factory=dict)
    budget_pressure: DistributedScalar | None = None


# ---------------------------------------------------------------------------
# P(z_pre|x) — GenAI baseline prior blocks
# ---------------------------------------------------------------------------


class ProcessingProfile(BaseModel):
    """Cognitive and behavioral ad-processing style (PersonaTemplate / GenAI)."""

    ad_tolerance: DistributedScalar
    price_sensitivity: DistributedScalar
    narrative_engagement: DistributedScalar | None = None
    persuasion_resistance: DistributedScalar | None = None


class BrandPrior(BaseModel):
    """Pre-exposure brand familiarity and recall priors.

    Sampled from the PersonaTemplate brand_prior_mean/std. All fields use
    status='assumed' with low confidence — there is no external empirical
    calibration of brand familiarity in the production pipeline.
    Constraint: unaided_recall.mean <= aided_recognition.mean.
    """

    familiarity: DistributedScalar
    unaided_recall: DistributedScalar
    aided_recognition: DistributedScalar
    consideration: DistributedScalar
    # Unipolar [0, 1]: 0 = no positive feeling (indifferent or actively negative),
    # 1 = maximally favorable. None when the brand prior is unknown — the
    # baseline_initializer falls back to baseline.default_favorability (~0.10).
    favorability: DistributedScalar | None = None


class LatentPreExposureState(BaseModel):
    """Latent psychological state before ad exposure — direct output of FR-09A."""

    brand_memory_accessibility: DistributedScalar | None = None
    consideration_readiness: DistributedScalar | None = None
    novel_brand_openness: DistributedScalar | None = None
    purchase_intent_pre: DistributedScalar | None = None


class SessionState(BaseModel):
    """Session-level behavioral priors (explicit simulation assumptions)."""

    session_length_tendency: DistributedScalar
    search_propensity: DistributedScalar
    click_propensity: DistributedScalar
    viewing_context: str | None = None
    avg_daily_video_sessions: float | None = Field(default=None, ge=0.0)


# ---------------------------------------------------------------------------
# FR-09B — twin-world sampling support
# ---------------------------------------------------------------------------


class BaselineDraw(BaseModel):
    """Sampled point values drawn from DistributedScalar distributions (FR-09B).

    These are the scalar values used to initialize agent_internal_state.
    Both twins in a pair share identical BaselineDraw values.
    """

    ad_tolerance: float | None = Field(default=None, ge=0.0, le=1.0)
    price_sensitivity: float | None = Field(default=None, ge=0.0, le=1.0)
    narrative_engagement: float | None = Field(default=None, ge=0.0, le=1.0)
    persuasion_resistance: float | None = Field(default=None, ge=0.0, le=1.0)
    familiarity: float | None = Field(default=None, ge=0.0, le=1.0)
    unaided_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    aided_recognition: float | None = Field(default=None, ge=0.0, le=1.0)
    consideration: float | None = Field(default=None, ge=0.0, le=1.0)
    favorability: float | None = Field(default=None, ge=0.0, le=1.0)
    purchase_intent_pre: float | None = Field(default=None, ge=0.0, le=1.0)
    search_propensity: float | None = Field(default=None, ge=0.0, le=1.0)
    click_propensity: float | None = Field(default=None, ge=0.0, le=1.0)
    session_length_tendency: float | None = Field(default=None, ge=0.0, le=1.0)


class WorldType(str, Enum):
    TREATMENT = "treatment"
    CONTROL = "control"


class TwinPairing(BaseModel):
    """Twin-world pair tracking (FR-09B)."""

    pair_id: str
    world_type: WorldType
    sibling_persona_id: str | None = None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class GenerationStage(str, Enum):
    SKELETON = "skeleton"
    POPULATION_SAMPLED = "population_sampled"
    BASELINE_PRIOR_GENERATED = "baseline_prior_generated"
    BASELINE_DRAWN = "baseline_drawn"
    TWIN_INITIALIZED = "twin_initialized"


class BaselineGenerationMethod(str, Enum):
    GENAI_ENSEMBLE = "genai_ensemble"
    GENAI_SINGLE = "genai_single"
    PUBLIC_PROXY_ONLY = "public_proxy_only"
    HARDCODED_HEURISTIC = "hardcoded_heuristic"


class Provenance(BaseModel):
    """Pipeline metadata for audit trail (§1.11.2) and reproducibility (§1.11.1)."""

    generation_stage: GenerationStage
    baseline_generation_method: BaselineGenerationMethod | None = None
    llm_model: str | None = None
    ensemble_seeds: list[int] = Field(default_factory=list)
    public_data_sources: list[str] = Field(default_factory=list)
    created_at: str | None = None
    seed: int | None = None


# ---------------------------------------------------------------------------
# Memory state
# ---------------------------------------------------------------------------


class MemoryState(BaseModel):
    """Agent memory state."""

    brand_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    ad_exposure_count: int = Field(default=0, ge=0)
    last_search_query: str = Field(default="")
    purchase_history: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Persona (FR-08)
# ---------------------------------------------------------------------------


class Persona(BaseModel):
    """State-variable-based persona definition.

    Acceptance criteria:
    - Can be stored as numeric/categorical attributes rather than natural language profiles
    - Supports intra-segment variation
    - Three-stage pipeline layers are clearly separated:
        segment_attrs / digital_profile  → P(x) demographic facts
        processing_profile / brand_prior / latent_pre_exposure_state → P(z_pre|x) GenAI priors
        agent_internal_state             → dynamic simulation state (FR-15)
    """

    persona_id: str = ""
    segment: str = ""

    # P(x) — demographic facts
    segment_attrs: SegmentAttrs
    digital_profile: DigitalProfile | None = None
    category_profile: CategoryProfile

    # P(z_pre|x) — GenAI baseline prior
    processing_profile: ProcessingProfile
    brand_prior: BrandPrior
    latent_pre_exposure_state: LatentPreExposureState | None = None

    # Session behavioral prior
    session_state: SessionState

    # FR-09B — twin-world sampling
    baseline_draw: BaselineDraw | None = None
    twin_pairing: TwinPairing | None = None

    # Dynamic state (initialized from baseline_draw, diverges per world)
    agent_internal_state: Any = Field(default=None)
    memory_state: MemoryState = Field(default_factory=MemoryState)

    # Audit trail
    provenance: Provenance

    def clone(self) -> Persona:
        """Return a full deep copy (for Twin generation, FR-09B)."""
        return self.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Convenience scalar properties (flat access for analysis/simulator)
    # ------------------------------------------------------------------

    @property
    def ad_tolerance(self) -> float:
        return self.processing_profile.ad_tolerance.mean

    @property
    def price_sensitivity(self) -> float:
        return self.processing_profile.price_sensitivity.mean

    @property
    def search_propensity(self) -> float:
        return self.session_state.search_propensity.mean

    @property
    def click_propensity(self) -> float:
        return self.session_state.click_propensity.mean

    @property
    def session_length_tendency(self) -> float:
        return self.session_state.session_length_tendency.mean

    @property
    def brand_familiarity(self) -> float:
        """Scalar brand familiarity (from brand_prior.familiarity.mean)."""
        return self.brand_prior.familiarity.mean

