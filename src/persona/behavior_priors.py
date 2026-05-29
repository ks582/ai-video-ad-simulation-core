"""Behavior prior providers for persona generation.

Decouples per-segment behavioral attribute means from the persona-template
defaults so that scenario-based sensitivity analysis can shift the assumption
band while preserving per-segment heterogeneity.

Reaction priors here are explicit simulation assumptions, not empirical
calibrations. See REACTION_PRIOR_DISCLOSURE for the user-facing wording.

This module also hosts the public API for Phase 1's LLM-elicited behavior
baseline pipeline:
  - `resolve_active_llm_elicited_snapshot(baselines_dir)` — active.json reader
  - `load_llm_elicited_snapshot(snapshot_path)` — snapshot loader
  - `LLMElicitedBehaviorPriorProvider` — provider class
  - `LLM_ELICITED_DISCLOSURE` — Phase-2-surfaced disclosure string

Shared validation constants (MAPPED_ATTRS, DEFAULT_BOUNDS, etc.) and
helpers (LLMElicitedSnapshotError, _reject_nonfinite, etc.) live in
`src/persona/_baseline_schema.py`; this module re-exports them for
backward-compatible imports from existing callers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from src.persona._baseline_schema import (
    DEFAULT_BOUNDS,
    LLM_ELICITED_ALLOWED_RANGES,
    LLMElicitedSnapshotError,
    MAPPED_ATTRS,
    SnapshotMeta,
    _LLM_ELICITED_MIN_WIDTH_FRACTION,
    _VERSION_PATTERN,
    _reject_nonfinite,
    validate_snapshot_invariants,
)

# Re-export tokens for backward-compatibility. Tools/tests importing these
# names from `src.persona.behavior_priors` continue to work.
__all__ = [
    # Re-exports from _baseline_schema (plan-mandated list, R10.7-R10.8)
    "DEFAULT_BOUNDS",
    "LLM_ELICITED_ALLOWED_RANGES",
    "LLMElicitedSnapshotError",
    "MAPPED_ATTRS",
    "SnapshotMeta",
    "_LLM_ELICITED_MIN_WIDTH_FRACTION",
    "_VERSION_PATTERN",
    # Module-local existing API
    "BehaviorPriorProvider",
    "BehaviorPriorSample",
    "NAS_CALIBRATION_DISCLOSURE",
    "PIPELINE_MISSION",
    "REACTION_PRIOR_DISCLOSURE",
    "ScenarioAssumptions",
    "ScenarioBehaviorPriorProvider",
    "SCENARIOS",
    "TemplateBehaviorPriorProvider",
    "TemplateLike",
    # Phase 1 additions
    "LLM_ELICITED_DISCLOSURE",
    "LLMElicitedBehaviorPriorProvider",
    "PROJECT_ROOT",
    "load_llm_elicited_snapshot",
    "resolve_active_llm_elicited_snapshot",
]


# Project root used by load_llm_elicited_snapshot for invariant 10 (prompt
# hash recompute). Resolves to the repo root in source checkouts and to
# site-packages root in wheel installs; per plan (Hatch force-include), the
# `prompts/behavior_baseline/` tree is co-located with this module in both
# layouts.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class TemplateLike(Protocol):
    """Structural protocol covering the persona-template fields this provider reads.

    Decouples this module from the concrete template class so it remains portable
    across deployment surfaces.
    """

    session_length_tendency: float
    search_propensity: float
    click_propensity: float
    session_length_tendency_std: float
    search_propensity_std: float
    click_propensity_std: float


@dataclass(frozen=True)
class BehaviorPriorSample:
    """One sampled set of behavioral assumption means.

    Spread (std) remains the template's responsibility — providers override
    means only.
    """

    session_length_tendency: float
    search_propensity: float
    click_propensity: float


def _map_to_band(
    v: float, source: tuple[float, float], target: tuple[float, float]
) -> float:
    """Linear z-mapping: position in source [0..1] -> same position in target band."""
    src_lo, src_hi = source
    if src_hi <= src_lo:
        raise ValueError(f"_map_to_band source has zero or negative width: {source}")
    z = max(0.0, min(1.0, (v - src_lo) / (src_hi - src_lo)))
    return target[0] + z * (target[1] - target[0])


@dataclass(frozen=True)
class ScenarioAssumptions:
    """Scenario assumption bands (NOT multipliers).

    Sampling: linear z-mapping from DEFAULT_BOUNDS source range into the
    scenario band, preserving per-segment relative ordering and proportional
    spacing. Clamping was rejected because under conservative search band
    (0.20-0.30), template defaults {0.30, 0.50, 0.60} would all collapse to
    0.30 — destroying segment heterogeneity.
    """

    session_length_tendency_range: tuple[float, float]
    search_propensity_range: tuple[float, float]
    click_propensity_range: tuple[float, float]
    rationale: str  # MANDATORY


# Scenario bands chosen to span and modestly extend beyond the current
# PersonaTemplate default range. NOT calibrated to any external benchmark.
# Source-of-truth is this file; changes require updating the rationale string.
# Ordering invariant (enforced by test):
#   conservative.upper <= base.lower <= base.upper <= optimistic.lower
SCENARIOS: dict[str, ScenarioAssumptions] = {
    "conservative": ScenarioAssumptions(
        session_length_tendency_range=(0.20, 0.30),
        search_propensity_range=(0.20, 0.30),
        click_propensity_range=(0.05, 0.15),
        rationale=(
            "Lower-bound assumption band. Sits at or below the current "
            "PersonaTemplate default range. No empirical grounding."
        ),
    ),
    "base": ScenarioAssumptions(
        session_length_tendency_range=(0.30, 0.55),
        search_propensity_range=(0.30, 0.55),
        click_propensity_range=(0.15, 0.30),
        rationale=(
            "Mid-range assumption band. Overlaps the current PersonaTemplate "
            "default range. No empirical grounding."
        ),
    ),
    "optimistic": ScenarioAssumptions(
        session_length_tendency_range=(0.55, 0.75),
        search_propensity_range=(0.55, 0.75),
        click_propensity_range=(0.30, 0.50),
        rationale=(
            "Upper-bound assumption band. Sits at or above the current "
            "PersonaTemplate default range. No empirical grounding."
        ),
    ),
}


class BehaviorPriorProvider(Protocol):
    """Protocol for behavioral assumption providers.

    Returns means; spread (std) is applied by the consumer using the template's
    *_std fields and the provider's std_scale multiplier.
    """

    def sample(
        self, rng: np.random.Generator, template: TemplateLike
    ) -> BehaviorPriorSample: ...

    def std_scale(self, attr: str) -> float:
        """Multiplier applied to template.X_std by the consumer when sampling.

        Template provider returns 1.0; Scenario provider compresses std
        proportionally to the scenario band width vs DEFAULT_BOUNDS source width.
        """
        ...

    @property
    def provenance_label(self) -> str: ...

    @property
    def scenario_name(self) -> str | None: ...


class TemplateBehaviorPriorProvider:
    """Returns the template's own MEANS verbatim. Default; identical to current behavior.

    Spread (std) is NOT returned by the provider; the consumer applies
    rng.normal(mean, template.X_std) AFTER calling sample().
    """

    @property
    def provenance_label(self) -> str:
        return "explicit simulation assumption (PersonaTemplate default)"

    @property
    def scenario_name(self) -> str | None:
        return None

    def sample(
        self, rng: np.random.Generator, template: TemplateLike
    ) -> BehaviorPriorSample:
        return BehaviorPriorSample(
            session_length_tendency=template.session_length_tendency,
            search_propensity=template.search_propensity,
            click_propensity=template.click_propensity,
        )

    def std_scale(self, attr: str) -> float:
        if attr not in MAPPED_ATTRS:
            raise ValueError(f"Unknown attribute: {attr!r}. Valid: {MAPPED_ATTRS}")
        return 1.0


class ScenarioBehaviorPriorProvider:
    """Linear z-mapping of template values into a scenario assumption band.

    Each segment's relative position in DEFAULT_BOUNDS is mapped to the same
    relative position in the scenario band. This preserves per-segment
    differentiation across all scenarios.
    """

    def __init__(self, scenario: str):
        # Validate scenario name BEFORE dict lookup (clearer error than KeyError)
        if scenario not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario!r}. Valid: {sorted(SCENARIOS)}"
            )
        self._scenario = scenario
        self._assumptions = SCENARIOS[scenario]
        # Construction-time validation.
        # Zero-width band (band[1] <= band[0]) is rejected because it would
        # degenerate std_scale to 0 and rng.normal(mean, 0) to a deterministic
        # constant. If a caller ever needs deterministic behavior, that must be
        # a distinct provider type, not a degenerate scenario band.
        for attr in MAPPED_ATTRS:
            band = getattr(self._assumptions, f"{attr}_range")
            if band[1] <= band[0]:
                raise ValueError(
                    f"Scenario {scenario!r} band for {attr} is zero-width or inverted: {band}"
                )
            src_lo, src_hi = DEFAULT_BOUNDS[attr]
            if src_hi <= src_lo:
                raise ValueError(
                    f"DEFAULT_BOUNDS for {attr} has zero or negative width: ({src_lo}, {src_hi})"
                )

    @property
    def provenance_label(self) -> str:
        return f"explicit simulation assumption (scenario={self._scenario})"

    @property
    def scenario_name(self) -> str:
        return self._scenario

    def sample(
        self, rng: np.random.Generator, template: TemplateLike
    ) -> BehaviorPriorSample:
        return BehaviorPriorSample(
            session_length_tendency=_map_to_band(
                template.session_length_tendency,
                DEFAULT_BOUNDS["session_length_tendency"],
                self._assumptions.session_length_tendency_range,
            ),
            search_propensity=_map_to_band(
                template.search_propensity,
                DEFAULT_BOUNDS["search_propensity"],
                self._assumptions.search_propensity_range,
            ),
            click_propensity=_map_to_band(
                template.click_propensity,
                DEFAULT_BOUNDS["click_propensity"],
                self._assumptions.click_propensity_range,
            ),
        )

    def std_scale(self, attr: str) -> float:
        """Scale template std proportionally to active band width.

        Prevents downstream rng.normal(mean, std) from blowing out the left
        tail under narrow bands.
        """
        if attr not in MAPPED_ATTRS:
            raise ValueError(f"Unknown attribute: {attr!r}. Valid: {MAPPED_ATTRS}")
        band = getattr(self._assumptions, f"{attr}_range")
        src = DEFAULT_BOUNDS[attr]
        return (band[1] - band[0]) / (src[1] - src[0])


REACTION_PRIOR_DISCLOSURE = (
    "Behavioral reaction metrics in this simulation — including skip rate, "
    "click-through, and attention indicators — are model assumptions, not "
    "empirical measurements. No independent benchmark calibrates these outputs "
    "to real YouTube audiences. Results support structured comparison between "
    "creative options; they do not forecast real-world campaign performance. "
    "Do not present these outputs to clients or stakeholders as performance "
    "predictions."
)

NAS_CALIBRATION_DISCLOSURE = (
    "Null-stimulus calibration removes LLM-side response bias. "
    "It does not ground reaction estimates in empirical data."
)

PIPELINE_MISSION = (
    "Structured decision-support tool for creative selection and hypothesis "
    "generation. Ranks options under explicit, uncalibrated assumptions; does "
    "not forecast real-world performance and has not been validated against "
    "observed campaign outcomes."
)


# ---------------------------------------------------------------------------
# Phase 1: LLM-elicited behavior baseline pipeline (public API)
# ---------------------------------------------------------------------------

LLM_ELICITED_DISCLOSURE = (
    "Behavioral reaction metrics in this simulation are model assumptions, "
    "not empirical measurements. The 'llm_elicited' scenario uses behavioral "
    "prior bands generated by a multi-model LLM ensemble. These elicited "
    "priors represent informed model opinion about plausible parameter "
    "ranges, not calibrated empirical estimates. Results support structured "
    "comparison between creative options; they do not forecast real-world "
    "campaign performance. Do not present these outputs to clients or "
    "stakeholders as performance predictions."
)


def resolve_active_llm_elicited_snapshot(baselines_dir: Path) -> Path:
    """Read active.json, validate version, return resolved snapshot path.

    Single source of truth for active.json validation. Hardened against
    malformed JSON (lists, numbers, NaN, missing keys, etc.). Does NOT verify
    the resulting snapshot path exists — that is the caller's job (see plan
    R4.7 / NH2 three-step pattern).

    Raises LLMElicitedSnapshotError on:
      - active.json missing or unreadable (FileNotFoundError or OSError)
      - active.json unparseable JSON or contains NaN/Infinity
      - top-level JSON is not a dict
      - version field absent / not a string / not matching _VERSION_PATTERN
    """
    active = baselines_dir / "active.json"
    try:
        raw = active.read_bytes()
    except FileNotFoundError as e:
        raise LLMElicitedSnapshotError(f"active.json missing: {e}") from e
    except OSError as e:
        # PermissionError, IsADirectoryError, etc.
        raise LLMElicitedSnapshotError(f"active.json unreadable: {e}") from e
    try:
        data = json.loads(raw, parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMElicitedSnapshotError(f"active.json unparseable: {e}") from e
    if not isinstance(data, dict):
        raise LLMElicitedSnapshotError(
            f"active.json must be a JSON object, got {type(data).__name__}"
        )
    version = data.get("version")
    if not isinstance(version, str) or not _VERSION_PATTERN.match(version):
        raise LLMElicitedSnapshotError(
            f"active.json version {version!r} fails regex — possible path traversal"
        )
    return (baselines_dir / f"{version}.json").resolve()


def load_llm_elicited_snapshot(
    snapshot_path: Path,
) -> tuple[ScenarioAssumptions, SnapshotMeta]:
    """Load and validate a frozen monthly baseline snapshot.

    Returns `(ScenarioAssumptions, SnapshotMeta)` on success. The
    `ScenarioAssumptions.rationale` is synthesized from snapshot metadata
    (version + prompt_hash + models + calibrated=false).

    Raises LLMElicitedSnapshotError on:
      - file missing or unreadable
      - JSON unparseable or contains NaN/Infinity
      - top-level not a dict
      - any of the 31 snapshot invariants from `_baseline_schema` fails

    The loader passes `snapshot_path` and `PROJECT_ROOT` so that all
    invariants — including invariant 10 (prompt-hash recompute) and
    invariant 30 (filename↔version match) — fire. There is no fallback path
    in Phase 1: snapshot validation failure raises hard-stop per plan B1.
    """
    try:
        raw = snapshot_path.read_bytes()
    except FileNotFoundError as e:
        raise LLMElicitedSnapshotError(f"snapshot file missing: {snapshot_path}: {e}") from e
    except OSError as e:
        raise LLMElicitedSnapshotError(
            f"snapshot file unreadable: {snapshot_path}: {e}"
        ) from e
    try:
        snapshot_raw = json.loads(raw, parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMElicitedSnapshotError(
            f"snapshot file unparseable: {snapshot_path}: {e}"
        ) from e

    violations = validate_snapshot_invariants(
        snapshot_raw,
        snapshot_path=snapshot_path,
        project_root=PROJECT_ROOT,
        active_snapshot=None,
        check_prompt_hash=True,
        check_version_not_newer=False,
        idempotent_resume_context=None,
    )
    if violations:
        head = "; ".join(
            f"[{v.code}/{v.category}] {v.message}" for v in violations[:3]
        )
        more = f" (+{len(violations) - 3} more)" if len(violations) > 3 else ""
        raise LLMElicitedSnapshotError(
            f"Snapshot {snapshot_path} failed invariants: {head}{more}"
        )

    # Per plan NH1.1: explicit cast (not asdict) — TypedDict constructor would
    # require enumerating every field, but here the snapshot dict has been
    # validated as conforming to SnapshotMeta, so cast is the correct narrowing.
    meta = cast(SnapshotMeta, snapshot_raw)

    # Synthesize rationale (plan R2.7). Use short prompt_hash prefix and the
    # successful_models list for human readability.
    ph_short = meta["prompt_hash"].split(":", 1)[-1][:12]
    models_csv = ",".join(meta["successful_models"])
    rationale = (
        f"LLM-elicited monthly baseline "
        f"(version={meta['version']}, "
        f"prompt_hash={ph_short}, "
        f"models=[{models_csv}], "
        f"calibrated=false)"
    )

    attrs = meta["attributes"]
    # `attrs[...][lo/hi]` is typed `object` from the nested `Mapping[str, object]`
    # in SnapshotMeta. `cast(float, ...)` narrows for mypy; the outer `float(...)`
    # coerces at runtime in case the JSON parser produced an int (e.g. `0`)
    # rather than a float, since `ScenarioAssumptions` expects `tuple[float, float]`.
    # Both layers are necessary; do NOT remove either as a "cleanup".
    assumptions = ScenarioAssumptions(
        session_length_tendency_range=(
            float(cast(float, attrs["session_length_tendency"]["lo"])),
            float(cast(float, attrs["session_length_tendency"]["hi"])),
        ),
        search_propensity_range=(
            float(cast(float, attrs["search_propensity"]["lo"])),
            float(cast(float, attrs["search_propensity"]["hi"])),
        ),
        click_propensity_range=(
            float(cast(float, attrs["click_propensity"]["lo"])),
            float(cast(float, attrs["click_propensity"]["hi"])),
        ),
        rationale=rationale,
    )
    return assumptions, meta


class LLMElicitedBehaviorPriorProvider:
    """Behavior prior provider backed by a frozen LLM-elicited snapshot.

    Independent class (NOT a subclass of ScenarioBehaviorPriorProvider) so it
    holds `self._assumptions: ScenarioAssumptions` directly without ever
    mutating the module-level `SCENARIOS` dict (plan NH1.2).

    Construct via the `from_snapshot(path)` classmethod factory. The factory
    delegates to `load_llm_elicited_snapshot()`; tests may also construct the
    provider directly when supplying a pre-validated `SnapshotMeta`.
    """

    def __init__(self, assumptions: ScenarioAssumptions, meta: SnapshotMeta) -> None:
        self._assumptions = assumptions
        self._meta = meta

    @classmethod
    def from_snapshot(cls, path: Path) -> "LLMElicitedBehaviorPriorProvider":
        """Factory: load + validate snapshot, return provider."""
        assumptions, meta = load_llm_elicited_snapshot(path)
        return cls(assumptions, meta)

    @property
    def provenance_label(self) -> str:
        m = self._meta
        ph_short = m["prompt_hash"].split(":", 1)[-1][:12]
        models_csv = ",".join(m["successful_models"])
        return (
            f"explicit simulation assumption (scenario=llm_elicited, "
            f"version={m['version']}, models=[{models_csv}], "
            f"prompt_hash={ph_short})"
        )

    @property
    def scenario_name(self) -> str:
        return "llm_elicited"

    @property
    def behavior_prior_version(self) -> str:
        return self._meta["version"]

    def sample(
        self, rng: np.random.Generator, template: TemplateLike
    ) -> BehaviorPriorSample:
        """Linear z-mapping into the snapshot's bands.

        Mirrors `ScenarioBehaviorPriorProvider.sample` (15 lines duplicated
        per plan NH1.2 to avoid SCENARIOS dict coupling).
        """
        return BehaviorPriorSample(
            session_length_tendency=_map_to_band(
                template.session_length_tendency,
                DEFAULT_BOUNDS["session_length_tendency"],
                self._assumptions.session_length_tendency_range,
            ),
            search_propensity=_map_to_band(
                template.search_propensity,
                DEFAULT_BOUNDS["search_propensity"],
                self._assumptions.search_propensity_range,
            ),
            click_propensity=_map_to_band(
                template.click_propensity,
                DEFAULT_BOUNDS["click_propensity"],
                self._assumptions.click_propensity_range,
            ),
        )

    def std_scale(self, attr: str) -> float:
        """Scale template std proportionally to active band width.

        Mirrors `ScenarioBehaviorPriorProvider.std_scale`.
        """
        if attr not in MAPPED_ATTRS:
            raise ValueError(f"Unknown attribute: {attr!r}. Valid: {MAPPED_ATTRS}")
        band = getattr(self._assumptions, f"{attr}_range")
        src = DEFAULT_BOUNDS[attr]
        return (band[1] - band[0]) / (src[1] - src[0])