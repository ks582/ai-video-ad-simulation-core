"""Shared validation constants and helpers for the LLM-elicited behavior baseline pipeline.

Owns all constants, TypedDicts, dataclasses, and validation logic shared by:
  - the loader (behavior_priors.load_llm_elicited_snapshot)
  - the validator script (scripts.validate_baseline_swap)
  - the generator script (scripts.generate_behavior_baseline)
  - the swap script (scripts.swap_baseline)

This module MUST NOT import behavior_priors. behavior_priors re-exports
public constants from here for backward compatibility.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, TypedDict, cast


# ---------------------------------------------------------------------------
# Behavioral attribute constants
# ---------------------------------------------------------------------------

MAPPED_ATTRS: tuple[str, ...] = (
    "session_length_tendency",
    "search_propensity",
    "click_propensity",
)


# Coordinate system for ScenarioBehaviorPriorProvider z-mapping AND the
# LLM-elicited width guard (invariant 16). Read-only mapping.
DEFAULT_BOUNDS: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "session_length_tendency": (0.30, 0.60),
        "search_propensity": (0.30, 0.60),
        "click_propensity": (0.10, 0.30),
    }
)


# Per-attribute outer bounds for LLM-elicited snapshots (invariant 15).
# Floors set below SCENARIOS["conservative"] lower bounds and ceilings above
# SCENARIOS["optimistic"] upper bounds, giving elicitation freedom while
# rejecting absurd values.
LLM_ELICITED_ALLOWED_RANGES: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "session_length_tendency": (0.10, 0.85),
        "search_propensity": (0.10, 0.85),
        "click_propensity": (0.03, 0.60),
    }
)


# Minimum band width as fraction of DEFAULT_BOUNDS width. Bands narrower
# than this collapse std_scale toward 0 and destroy per-persona heterogeneity.
_LLM_ELICITED_MIN_WIDTH_FRACTION = 0.25

# A band exactly at the minimum passes by design (the check is strict `<`), but
# float arithmetic can leave it a few ULPs under min_width; treat within-tol as
# equal so a boundary-width band is not spuriously rejected. Shared by the
# generator's pre-write check so generate and validate agree.
_BAND_WIDTH_REL_TOL = 1e-9
_BAND_WIDTH_ABS_TOL = 1e-12


# Envelope for invariant 17 (scenario_envelope_overlap). Numerically equal to
# SCENARIOS["conservative"].lo and SCENARIOS["optimistic"].hi, defined here
# to avoid importing SCENARIOS from behavior_priors. Drift detected by
# test_envelope_constants_match_scenarios meta-test.
_LLM_ELICITED_SCENARIO_ENVELOPE: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "session_length_tendency": (0.20, 0.75),
        "search_propensity": (0.20, 0.75),
        "click_propensity": (0.05, 0.50),
    }
)


# ---------------------------------------------------------------------------
# Path / version patterns (path-traversal guards)
# ---------------------------------------------------------------------------

# YYYY-MM where MM is 01-12. Rejects 2026-13, 2026-00, "../etc/passwd", etc.
_VERSION_PATTERN: re.Pattern[str] = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

# Prompt directory name pattern; rejects "../", "v0", etc.
_PROMPT_VERSION_PATTERN: re.Pattern[str] = re.compile(r"^v[1-9]\d*$")


# ---------------------------------------------------------------------------
# Exception type
# ---------------------------------------------------------------------------


class LLMElicitedSnapshotError(ValueError):
    """Raised when a snapshot fails validation on load. No fallback path."""


# ---------------------------------------------------------------------------
# Allowlists for failed_models normalization
# ---------------------------------------------------------------------------

ALLOWED_PROVIDERS: frozenset[str] = frozenset({"google", "anthropic", "openai", "unknown"})

ALLOWED_PUBLIC_REASONS: frozenset[str] = frozenset(
    {
        "timeout",
        "rate_limit",
        "schema_validation_failed",
        "parse_error",
        "empty_response",
        "api_error",
    }
)


# ---------------------------------------------------------------------------
# TypedDicts and dataclasses
# ---------------------------------------------------------------------------


class FailedModelRecord(TypedDict, total=True):
    """Per-model failure record stored in snapshot.failed_models[model_id]."""

    error_type: str
    provider: str  # in ALLOWED_PROVIDERS
    public_reason: str  # in ALLOWED_PUBLIC_REASONS


class SnapshotMeta(TypedDict, total=True):
    """Validated snapshot top-level dict, returned by load_llm_elicited_snapshot."""

    schema_version: int
    version: str
    generated_at: str
    prompt_hash: str
    prompt_version: str
    requested_models: list[str]
    successful_models: list[str]
    failed_models: Mapping[str, FailedModelRecord]
    calibrated: bool
    scenario: str
    ensemble_method: str
    attributes: Mapping[str, Mapping[str, object]]
    per_model_raw: Mapping[str, Mapping[str, Mapping[str, object]]]
    cross_attribute_sanity: Mapping[str, bool]


@dataclass(frozen=True)
class BaselineViolation:
    """Structured invariant violation. Internal type returned by
    validate_snapshot_invariants. NOT JSON-serializable; converted to
    ViolationRecord (TypedDict) by run_validation before report emission.

    code maps directly to script exit codes:
      1 - range / allowed_range / magnitude / drift_* / version_not_newer
      2 - missing_pending_file / missing_active_pointer / missing_active_snapshot
          / missing_prompt_file
      3 - schema / type / derived / prompt_hash / filename / not_finite
          / bootstrap_with_existing_active / active_pointer_malformed
          / active_snapshot_malformed / scenario_envelope_overlap
          / prompt_version_invalid / prompt_schema_type_error
          / malformed_report
      5 - disjoint_bands
      6 - insufficient_ensemble
      7 - band_too_narrow
      8 - collision_different_content / downgrade_attempt
    """

    code: int
    category: str
    message: str


class ViolationRecord(TypedDict, total=True):
    """JSON-serializable representation of a BaselineViolation.

    Written into ValidationReport.violations. Constructed via TypedDict
    constructor call from BaselineViolation, NOT via dataclasses.asdict
    (asdict returns dict[str, Any] which fails mypy-strict assignment).
    """

    code: int
    category: str
    message: str


class ValidationReport(TypedDict, total=True):
    """JSON-serializable validator output.

    Nullable fields use `str | None` so a single flat shape covers normal
    runs, bootstrap (all active_* None), missing/malformed-active variants.
    Every nullable access requires a None-check at the call site.
    """

    exit_code: int
    violations: list[ViolationRecord]

    pending_path: str | None
    pending_version: str | None
    pending_sha256: str | None

    active_pointer_path: str | None
    active_pointer_sha256: str | None
    active_snapshot_path: str | None
    active_snapshot_sha256: str | None
    active_version: str | None


# ---------------------------------------------------------------------------
# Category -> exit-code mapping and priority
# ---------------------------------------------------------------------------

_CATEGORY_TO_CODE_RAW: dict[str, int] = {
    # code 1 - bypassable drift/magnitude
    "magnitude": 1,
    "drift_model_set": 1,
    "drift_ensemble": 1,
    "drift_prompt": 1,
    # code 1 - non-bypassable
    "range": 1,
    "allowed_range": 1,
    "version_not_newer": 1,
    # code 2 - missing files
    "missing_pending_file": 2,
    "missing_active_pointer": 2,
    "missing_active_snapshot": 2,
    "missing_prompt_file": 2,
    # code 3 - schema / type / malformed
    "schema": 3,
    "unknown_key": 3,
    "type": 3,
    "cross_attr_sanity": 3,
    "derived_metadata_mismatch": 3,
    "prompt_hash_mismatch": 3,
    "version_filename_mismatch": 3,
    "not_finite": 3,
    "generated_at_invalid": 3,
    "bootstrap_with_existing_active": 3,
    "active_pointer_malformed": 3,
    "active_snapshot_malformed": 3,
    "prompt_version_invalid": 3,
    "prompt_schema_type_error": 3,
    "scenario_envelope_overlap": 3,
    "malformed_report": 3,
    # code 5 - disjoint bands
    "disjoint_bands": 5,
    # code 6 - insufficient ensemble
    "insufficient_ensemble": 6,
    # code 7 - band too narrow
    "band_too_narrow": 7,
    # code 8 - immutability violations
    "collision_different_content": 8,
    "downgrade_attempt": 8,
}

CATEGORY_TO_CODE: Mapping[str, int] = MappingProxyType(_CATEGORY_TO_CODE_RAW)


# Priority dispatch order for multi-violation exit code selection.
# Higher-priority code wins; code 0 is implicit (only when violations == []).
PRIORITY_ORDER: tuple[int, ...] = (3, 8, 5, 6, 7, 2, 1)


ALLOWED_EXIT_CODES: frozenset[int] = frozenset({0, 1, 2, 3, 5, 6, 7, 8})


# Drift categories bypassable by --force (code 1 only).
_FORCE_BYPASSABLE_CATEGORIES: frozenset[str] = frozenset(
    {"magnitude", "drift_model_set", "drift_ensemble", "drift_prompt"}
)


# Magnitude-shift threshold (plan "Drift Rules"). Phase 1 fires `magnitude`
# when |pending.attributes[attr].mean - active.attributes[attr].mean| exceeds
# this absolute threshold for ANY attribute. The default is conservative
# relative to band widths (e.g., click DEFAULT_BOUNDS width is 0.20, so 0.10
# is half-width). Tune here after observing real month-to-month variation.
# Bypassable via swap --force; never bypassed by --allow-* flags (those cover
# drift_model_set / drift_ensemble / drift_prompt only).
_MAGNITUDE_MEAN_SHIFT_THRESHOLD: float = 0.10


# ---------------------------------------------------------------------------
# JSON parse guards
# ---------------------------------------------------------------------------


def _reject_nonfinite(value: str) -> float:
    """parse_constant handler for json.loads.

    Python 3.11 stdlib json ACCEPTS NaN, Infinity, -Infinity by default.
    This handler rejects them explicitly so the entire pipeline fails closed
    on non-finite numeric JSON literals. Imported by generator/loader/swap.
    """
    raise ValueError(f"Non-finite JSON literal: {value}")


# ---------------------------------------------------------------------------
# Prompt path helpers
# ---------------------------------------------------------------------------


def _resolve_prompt_dir(project_root: Path, prompt_version: str) -> Path:
    """Validate prompt_version and resolve its directory under prompts/behavior_baseline/.

    Raises LLMElicitedSnapshotError on:
      - prompt_version fails _PROMPT_VERSION_PATTERN
      - resolved path escapes prompts/behavior_baseline/
      - resolved path is not an existing directory
    """
    if not _PROMPT_VERSION_PATTERN.match(prompt_version):
        raise LLMElicitedSnapshotError(
            f"prompt_version {prompt_version!r} fails regex"
        )
    root = (project_root / "prompts" / "behavior_baseline").resolve()
    candidate = (root / prompt_version).resolve()
    if root != candidate and root not in candidate.parents:
        raise LLMElicitedSnapshotError(
            f"prompt_version {prompt_version!r} escapes prompts dir"
        )
    if not candidate.is_dir():
        raise LLMElicitedSnapshotError(f"prompt_version dir not found: {candidate}")
    return candidate


def _load_output_schema(prompt_dir: Path) -> dict[str, object]:
    """Load output_schema.json, parse with NaN guard, verify top-level dict.

    Caller pattern (mandatory): callers MUST check
    `(prompt_dir / "output_schema.json").exists()` FIRST and emit
    BaselineViolation(code=2, category="missing_prompt_file") if missing.
    Only on the else branch should this function be called; failures here
    map to BaselineViolation(code=3, category="prompt_schema_type_error").
    A single try/except wrap around this call collapses the two categories.
    """
    schema_path = prompt_dir / "output_schema.json"
    try:
        raw = schema_path.read_bytes()
    except FileNotFoundError as e:
        # Caller should have checked exists() first; if we reach here it is a
        # late-binding race. Map to prompt_schema_type_error so caller can
        # decide which branch they failed to take.
        raise LLMElicitedSnapshotError(f"output_schema.json missing: {e}") from e
    except OSError as e:
        # IsADirectoryError, PermissionError, etc. — wrap so caller can map
        # to prompt_schema_type_error (code 3) per plan.
        raise LLMElicitedSnapshotError(f"output_schema.json unreadable: {e}") from e
    try:
        parsed = json.loads(raw, parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMElicitedSnapshotError(f"output_schema.json unparseable: {e}") from e
    if not isinstance(parsed, dict):
        raise LLMElicitedSnapshotError(
            f"output_schema.json top-level is {type(parsed).__name__}, expected dict"
        )
    return parsed


def _read_text_file(path: Path) -> str:
    """Read a UTF-8 text file or raise LLMElicitedSnapshotError on any I/O error."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise LLMElicitedSnapshotError(f"prompt file missing: {path}") from e
    except OSError as e:
        # IsADirectoryError, PermissionError, etc.
        raise LLMElicitedSnapshotError(f"prompt file unreadable: {path}: {e}") from e


def _render_prompt_bundle(
    project_root: Path, prompt_version: str
) -> tuple[str, str, Path]:
    """Render the prompt bundle for hashing AND for sending to the LLM.

    Returns (system_text, rendered_user_text, prompt_dir). The rendered_user_text
    iterates MAPPED_ATTRS in canonical order, substitutes per-attribute slots
    from attribute_descriptions.json + DEFAULT_BOUNDS, and embeds the canonical
    JSON of output_schema.json INLINE. The hash target is exactly
    system_text + rendered_user_text via _compute_prompt_hash.

    Raises LLMElicitedSnapshotError if system.txt, user.txt, or
    attribute_descriptions.json are missing.
    """
    prompt_dir = _resolve_prompt_dir(project_root, prompt_version)
    system_text = _read_text_file(prompt_dir / "system.txt")
    user_template = _read_text_file(prompt_dir / "user.txt")

    # attribute_descriptions.json: top-level dict mapping attr -> description.
    descriptions_path = prompt_dir / "attribute_descriptions.json"
    try:
        descriptions_raw = descriptions_path.read_bytes()
    except FileNotFoundError as e:
        raise LLMElicitedSnapshotError(
            f"prompt file missing: {descriptions_path}"
        ) from e
    except OSError as e:
        raise LLMElicitedSnapshotError(
            f"prompt file unreadable: {descriptions_path}: {e}"
        ) from e
    try:
        descriptions = json.loads(descriptions_raw, parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMElicitedSnapshotError(
            f"attribute_descriptions.json unparseable: {e}"
        ) from e
    if not isinstance(descriptions, dict):
        raise LLMElicitedSnapshotError(
            "attribute_descriptions.json top-level is not a dict"
        )

    # output_schema.json: embedded inline as canonical JSON (single source).
    # B-CR Caller pattern: structural .exists() gate before _load_output_schema
    # to keep missing_prompt_file (code 2) distinct from prompt_schema_type_error
    # (code 3). Single try/except wrapping is forbidden.
    schema_path = prompt_dir / "output_schema.json"
    if not schema_path.exists():
        raise LLMElicitedSnapshotError(f"prompt file missing: {schema_path}")
    schema = _load_output_schema(prompt_dir)
    canonical_schema = json.dumps(schema, sort_keys=True, separators=(",", ":"))

    # Build rendered_user_text by iterating MAPPED_ATTRS in canonical order.
    rendered_parts: list[str] = [user_template, "\n\n--- attributes ---\n"]
    for attr in MAPPED_ATTRS:
        description = descriptions.get(attr, "")
        if not isinstance(description, str):
            raise LLMElicitedSnapshotError(
                f"attribute_descriptions.json[{attr!r}] is not a string"
            )
        src_lo, src_hi = DEFAULT_BOUNDS[attr]
        rendered_parts.append(
            f"\nattribute_name: {attr}\n"
            f"attribute_description: {description}\n"
            f"source_range_lo: {src_lo}\n"
            f"source_range_hi: {src_hi}\n"
        )
    rendered_parts.append("\n--- output schema (return ONLY a JSON object matching this) ---\n")
    rendered_parts.append(canonical_schema)
    rendered_parts.append("\n")
    rendered_user_text = "".join(rendered_parts)

    return system_text, rendered_user_text, prompt_dir


def _compute_prompt_hash(system_text: str, rendered_user_text: str) -> str:
    """Compute SHA-256 prompt hash via canonical JSON wrapping.

    Uses json.dumps with sorted keys so the system/user boundary is
    unambiguous. The schema is already embedded inside rendered_user_text
    via _render_prompt_bundle, so it is hashed exactly once (transitively).
    Format: "sha256:<64 lowercase hex chars>".
    """
    canonical = json.dumps(
        {"system": system_text, "user": rendered_user_text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# Snapshot invariant validation (31 invariants)
# ---------------------------------------------------------------------------


_SNAPSHOT_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "version",
        "generated_at",
        "prompt_hash",
        "prompt_version",
        "requested_models",
        "successful_models",
        "failed_models",
        "calibrated",
        "scenario",
        "ensemble_method",
        "attributes",
        "per_model_raw",
        "cross_attribute_sanity",
    }
)


_ATTRIBUTE_OBJECT_KEYS: frozenset[str] = frozenset(
    {"lo", "hi", "mean", "plausible_range", "std", "rationale"}
)


_PER_MODEL_ATTR_KEYS: frozenset[str] = frozenset(
    {"lo", "hi", "mean", "std", "rationale"}
)


_FAILED_MODEL_KEYS: frozenset[str] = frozenset(
    {"error_type", "provider", "public_reason"}
)


_CROSS_ATTRIBUTE_SANITY_KEYS: frozenset[str] = frozenset(
    {"click_band_entirely_below_search_band", "session_length_widest_band"}
)


_DERIVED_TOLERANCE = 1e-9


def _is_finite_number(v: object) -> bool:
    """True if v is a finite int or float (not bool — bool is int subclass)."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return v == v and v not in (float("inf"), float("-inf"))
    return False


def validate_snapshot_invariants(
    snapshot: object,
    *,
    snapshot_path: Path | None = None,
    project_root: Path | None = None,
    active_snapshot: Mapping[str, object] | None = None,
    check_prompt_hash: bool = False,
    check_version_not_newer: bool = False,
    idempotent_resume_context: Mapping[str, bool] | None = None,
) -> list[BaselineViolation]:
    """Run all 31 invariants. Return list of BaselineViolation (empty = valid).

    Accepts `object` (not bare `dict`) because JSON parsing can yield non-object
    top-level values; invariant 1 fires when snapshot is not a dict.

    Used by:
      - loader (raises LLMElicitedSnapshotError if non-empty)
      - validator script (priority-dispatch exit code from PRIORITY_ORDER)
      - generator post-write self-check (FATAL: deletes tmp + non-zero exit)
    """
    violations: list[BaselineViolation] = []

    def add(code: int, category: str, message: str) -> None:
        violations.append(BaselineViolation(code=code, category=category, message=message))

    # Invariant 1: Snapshot is a JSON object.
    if not isinstance(snapshot, dict):
        add(3, "schema", f"snapshot top-level is {type(snapshot).__name__}, expected dict")
        return violations  # cannot continue without dict access

    # Invariant 2: Top-level keys exactly match; no legacy `models` field.
    actual_keys = set(snapshot.keys())
    if "models" in actual_keys:
        add(3, "unknown_key", "legacy 'models' field present; use requested/successful/failed_models")
    extra = actual_keys - _SNAPSHOT_TOP_LEVEL_KEYS
    missing = _SNAPSHOT_TOP_LEVEL_KEYS - actual_keys
    if extra:
        add(3, "unknown_key", f"unexpected top-level keys: {sorted(extra)}")
    if missing:
        add(3, "schema", f"missing required top-level keys: {sorted(missing)}")
    if missing:
        # Without the missing keys we cannot continue most invariants.
        return violations

    # Invariant 3: schema_version == 1
    if snapshot.get("schema_version") != 1:
        add(3, "schema", f"schema_version must be 1, got {snapshot.get('schema_version')!r}")

    # Invariant 4: calibrated is False
    if snapshot.get("calibrated") is not False:
        add(3, "schema", f"calibrated must be False, got {snapshot.get('calibrated')!r}")

    # Invariant 5: scenario == "llm_elicited"
    if snapshot.get("scenario") != "llm_elicited":
        add(3, "schema", f"scenario must be 'llm_elicited', got {snapshot.get('scenario')!r}")

    # Invariant 6: version is a string matching _VERSION_PATTERN
    version = snapshot.get("version")
    version_valid = isinstance(version, str) and bool(_VERSION_PATTERN.match(version))
    if not version_valid:
        add(3, "schema", f"version {version!r} fails _VERSION_PATTERN")

    # Invariant 7: generated_at is UTC ISO-8601 and year-month equals version
    generated_at = snapshot.get("generated_at")
    if not isinstance(generated_at, str):
        add(3, "generated_at_invalid", f"generated_at must be str, got {type(generated_at).__name__}")
    else:
        # Accept "...Z" or "+00:00". Year-month must match version.
        ga_year_month = _extract_utc_year_month(generated_at)
        if ga_year_month is None:
            add(3, "generated_at_invalid", f"generated_at {generated_at!r} not UTC ISO-8601")
        elif version_valid and isinstance(version, str) and ga_year_month != version:
            add(
                3,
                "generated_at_invalid",
                f"generated_at year-month {ga_year_month!r} != version {version!r}",
            )

    # Invariant 8: prompt_version matches pattern, dir exists under prompts/behavior_baseline/
    prompt_version = snapshot.get("prompt_version")
    pv_valid = isinstance(prompt_version, str) and bool(
        _PROMPT_VERSION_PATTERN.match(prompt_version)
    )
    if not pv_valid:
        add(3, "prompt_version_invalid", f"prompt_version {prompt_version!r} fails regex")
    elif project_root is not None and isinstance(prompt_version, str):
        try:
            _resolve_prompt_dir(project_root, prompt_version)
        except LLMElicitedSnapshotError as e:
            add(3, "prompt_version_invalid", str(e))

    # Invariant 9: prompt_hash format
    prompt_hash = snapshot.get("prompt_hash")
    if not (isinstance(prompt_hash, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", prompt_hash)):
        add(3, "schema", f"prompt_hash format invalid: {prompt_hash!r}")

    # Invariant 10: prompt_hash recompute (when check_prompt_hash=True)
    # B-CR caller pattern: structural .exists() check before _load_output_schema.
    # NO string-match on exception message — that's the category-collapse the
    # plan forbids. Two-stage: resolve dir, check each prompt file exists, then
    # load (any exception inside load → prompt_schema_type_error).
    if (
        check_prompt_hash
        and project_root is not None
        and isinstance(prompt_version, str)
        and pv_valid
        and isinstance(prompt_hash, str)
    ):
        # Stage 1: resolve prompt_dir (already validated by invariant 8, but
        # re-resolve here to get the Path safely)
        try:
            prompt_dir_for_hash = _resolve_prompt_dir(project_root, prompt_version)
        except LLMElicitedSnapshotError as e:
            add(3, "prompt_version_invalid", str(e))
        else:
            # Stage 2: structural existence check for each prompt file (code 2).
            required_files = ("system.txt", "user.txt", "attribute_descriptions.json", "output_schema.json")
            missing = [f for f in required_files if not (prompt_dir_for_hash / f).exists()]
            if missing:
                add(
                    2,
                    "missing_prompt_file",
                    f"prompt file(s) missing: {sorted(missing)} under {prompt_dir_for_hash}",
                )
            else:
                # Stage 3: render + load schema; any failure HERE is malformed (code 3).
                try:
                    sys_text, user_text, _ = _render_prompt_bundle(project_root, prompt_version)
                    recomputed = _compute_prompt_hash(sys_text, user_text)
                    if recomputed != prompt_hash:
                        add(
                            3,
                            "prompt_hash_mismatch",
                            f"recomputed {recomputed[:19]}... != stored {prompt_hash[:19]}...",
                        )
                except LLMElicitedSnapshotError as e:
                    add(3, "prompt_schema_type_error", str(e))

    # Invariant 11: ensemble_method
    if snapshot.get("ensemble_method") != "intersection_of_bands":
        add(
            3,
            "schema",
            f"ensemble_method must be 'intersection_of_bands', got {snapshot.get('ensemble_method')!r}",
        )

    # Invariants 12-18, 19-20: attributes block validation
    attributes = snapshot.get("attributes")
    attributes_ok = isinstance(attributes, dict) and set(attributes.keys()) == set(MAPPED_ATTRS)
    if not attributes_ok:
        add(3, "schema", f"attributes keys must be {sorted(MAPPED_ATTRS)}")

    final_bands: dict[str, tuple[float, float]] = {}
    if attributes_ok and isinstance(attributes, dict):
        for attr in MAPPED_ATTRS:
            attr_obj = attributes[attr]
            if not isinstance(attr_obj, dict):
                add(3, "type", f"attributes[{attr!r}] must be dict")
                continue
            if set(attr_obj.keys()) != _ATTRIBUTE_OBJECT_KEYS:
                add(
                    3,
                    "schema",
                    f"attributes[{attr!r}] keys must be {sorted(_ATTRIBUTE_OBJECT_KEYS)}",
                )
                continue
            lo = attr_obj["lo"]
            hi = attr_obj["hi"]
            mean = attr_obj["mean"]
            std = attr_obj["std"]
            plausible_range = attr_obj["plausible_range"]
            rationale = attr_obj["rationale"]

            # Invariant 13 numeric type + finite + non-empty rationale
            if not (_is_finite_number(lo) and _is_finite_number(hi)):
                add(3, "not_finite", f"attributes[{attr!r}].lo/hi not finite")
                continue
            if not (_is_finite_number(mean) and _is_finite_number(std)):
                add(3, "not_finite", f"attributes[{attr!r}].mean/std not finite")
            if not (isinstance(rationale, str) and rationale.strip()):
                add(3, "schema", f"attributes[{attr!r}].rationale is empty")

            lo_f = float(lo)  # type: ignore[arg-type]
            hi_f = float(hi)  # type: ignore[arg-type]
            mean_f = float(mean) if _is_finite_number(mean) else 0.0  # type: ignore[arg-type]
            std_f = float(std) if _is_finite_number(std) else 0.0  # type: ignore[arg-type]

            # Invariant 14: lo < hi
            if not (lo_f < hi_f):
                add(3, "schema", f"attributes[{attr!r}].lo {lo_f} >= hi {hi_f}")
                continue
            final_bands[attr] = (lo_f, hi_f)

            # Invariant 15: boundary in LLM_ELICITED_ALLOWED_RANGES
            allowed_lo, allowed_hi = LLM_ELICITED_ALLOWED_RANGES[attr]
            if not (allowed_lo <= lo_f and hi_f <= allowed_hi):
                add(
                    1,
                    "allowed_range",
                    f"attributes[{attr!r}] band ({lo_f}, {hi_f}) outside {(allowed_lo, allowed_hi)}",
                )

            # Invariant 16: width >= min_width
            src_lo, src_hi = DEFAULT_BOUNDS[attr]
            min_width = (src_hi - src_lo) * _LLM_ELICITED_MIN_WIDTH_FRACTION
            width = hi_f - lo_f
            if width < min_width and not math.isclose(
                width, min_width, rel_tol=_BAND_WIDTH_REL_TOL, abs_tol=_BAND_WIDTH_ABS_TOL
            ):
                add(
                    7,
                    "band_too_narrow",
                    f"attributes[{attr!r}] width {width} < min {min_width}",
                )

            # Invariant 17: scenario envelope overlap
            env_lo, env_hi = _LLM_ELICITED_SCENARIO_ENVELOPE[attr]
            if hi_f < env_lo or lo_f > env_hi:
                add(
                    3,
                    "scenario_envelope_overlap",
                    f"attributes[{attr!r}] band ({lo_f}, {hi_f}) does not overlap envelope {(env_lo, env_hi)}",
                )

            # Invariant 18: derived metadata
            expected_mean = (lo_f + hi_f) / 2
            expected_std = (hi_f - lo_f) / 4
            if _is_finite_number(mean) and abs(mean_f - expected_mean) > _DERIVED_TOLERANCE:
                add(
                    3,
                    "derived_metadata_mismatch",
                    f"attributes[{attr!r}].mean {mean_f} != (lo+hi)/2 {expected_mean}",
                )
            if _is_finite_number(std) and abs(std_f - expected_std) > _DERIVED_TOLERANCE:
                add(
                    3,
                    "derived_metadata_mismatch",
                    f"attributes[{attr!r}].std {std_f} != (hi-lo)/4 {expected_std}",
                )
            if not (
                isinstance(plausible_range, list)
                and len(plausible_range) == 2
                and _is_finite_number(plausible_range[0])
                and _is_finite_number(plausible_range[1])
                and abs(float(plausible_range[0]) - lo_f) < _DERIVED_TOLERANCE  # type: ignore[arg-type]
                and abs(float(plausible_range[1]) - hi_f) < _DERIVED_TOLERANCE  # type: ignore[arg-type]
            ):
                add(
                    3,
                    "derived_metadata_mismatch",
                    f"attributes[{attr!r}].plausible_range != [lo, hi]",
                )

    # Invariants 19, 20: cross_attribute_sanity
    cas = snapshot.get("cross_attribute_sanity")
    if not isinstance(cas, dict) or set(cas.keys()) != _CROSS_ATTRIBUTE_SANITY_KEYS:
        add(
            3,
            "cross_attr_sanity",
            f"cross_attribute_sanity keys must be {sorted(_CROSS_ATTRIBUTE_SANITY_KEYS)}",
        )
    elif not all(isinstance(v, bool) for v in cas.values()):
        add(3, "type", "cross_attribute_sanity values must be bool")
    elif len(final_bands) == 3:
        click = final_bands["click_propensity"]
        search = final_bands["search_propensity"]
        session = final_bands["session_length_tendency"]
        expected_click_below = click[1] < search[0]
        widths = {
            "session_length_tendency": session[1] - session[0],
            "search_propensity": search[1] - search[0],
            "click_propensity": click[1] - click[0],
        }
        expected_widest = widths["session_length_tendency"] >= max(
            widths["search_propensity"], widths["click_propensity"]
        )
        if cas["click_band_entirely_below_search_band"] != expected_click_below:
            add(
                3,
                "cross_attr_sanity",
                "cross_attribute_sanity.click_band_entirely_below_search_band does not match recomputation",
            )
        if cas["session_length_widest_band"] != expected_widest:
            add(
                3,
                "cross_attr_sanity",
                "cross_attribute_sanity.session_length_widest_band does not match recomputation",
            )

    # Invariants 21-26: model lists
    requested = snapshot.get("requested_models")
    successful = snapshot.get("successful_models")
    failed = snapshot.get("failed_models")

    if not _is_list_of_nonempty_strings(requested):
        add(3, "type", "requested_models must be list[non-empty str]")
        requested_list: list[str] = []
    else:
        requested_list = list(requested)  # type: ignore[arg-type]

    if not _is_list_of_nonempty_strings(successful):
        add(3, "type", "successful_models must be list[non-empty str]")
        successful_list: list[str] = []
    else:
        successful_list = list(successful)  # type: ignore[arg-type]

    if not isinstance(failed, dict) or not all(
        isinstance(k, str) and k for k in failed.keys()
    ):
        add(3, "type", "failed_models must be dict[non-empty str, ...]")
        failed_dict: dict[str, object] = {}
    else:
        failed_dict = dict(failed)

    # 21: requested unique
    if len(requested_list) != len(set(requested_list)):
        add(3, "schema", "requested_models has duplicates")
    # 22: successful unique + ordered by requested
    if len(successful_list) != len(set(successful_list)):
        add(3, "schema", "successful_models has duplicates")
    if requested_list and successful_list:
        ranks = [requested_list.index(m) for m in successful_list if m in requested_list]
        if ranks != sorted(ranks):
            add(
                3,
                "schema",
                "successful_models order does not follow requested_models",
            )
    # 23: successful ∩ failed.keys() == ∅
    if set(successful_list) & set(failed_dict.keys()):
        add(3, "schema", "successful_models and failed_models overlap")
    # 24: requested == successful ∪ failed
    if set(requested_list) != set(successful_list) | set(failed_dict.keys()):
        add(
            3,
            "schema",
            "set(requested_models) != set(successful_models) | set(failed_models)",
        )
    # 25: |successful| >= 2
    if len(successful_list) < 2:
        add(
            6,
            "insufficient_ensemble",
            f"successful_models has {len(successful_list)} entries; require >= 2",
        )

    # Invariant 26: per_model_raw.keys() == set(successful_models)
    per_model_raw = snapshot.get("per_model_raw")
    if not isinstance(per_model_raw, dict):
        add(3, "type", "per_model_raw must be dict")
        per_model_raw_dict: dict[str, object] = {}
    else:
        per_model_raw_dict = dict(per_model_raw)
    if set(per_model_raw_dict.keys()) != set(successful_list):
        add(3, "schema", "per_model_raw keys must equal successful_models")

    # Invariants 27, 28: per_model_raw structure + finite
    for model_id, model_attrs in per_model_raw_dict.items():
        if not isinstance(model_attrs, dict):
            add(3, "type", f"per_model_raw[{model_id!r}] must be dict")
            continue
        if set(model_attrs.keys()) != set(MAPPED_ATTRS):
            add(
                3,
                "schema",
                f"per_model_raw[{model_id!r}] attribute keys must be {sorted(MAPPED_ATTRS)}",
            )
            continue
        for attr, attr_payload in model_attrs.items():
            if not isinstance(attr_payload, dict):
                add(3, "type", f"per_model_raw[{model_id!r}][{attr!r}] must be dict")
                continue
            if set(attr_payload.keys()) != _PER_MODEL_ATTR_KEYS:
                add(
                    3,
                    "schema",
                    f"per_model_raw[{model_id!r}][{attr!r}] keys must be {sorted(_PER_MODEL_ATTR_KEYS)}",
                )
                continue
            pm_lo = attr_payload["lo"]
            pm_hi = attr_payload["hi"]
            pm_mean = attr_payload["mean"]
            pm_std = attr_payload["std"]
            pm_rationale = attr_payload["rationale"]
            if not all(
                _is_finite_number(v) for v in (pm_lo, pm_hi, pm_mean, pm_std)
            ):
                add(3, "not_finite", f"per_model_raw[{model_id!r}][{attr!r}] numerics not finite")
                continue
            if not (float(pm_lo) < float(pm_hi)):  # type: ignore[arg-type]
                add(
                    3,
                    "schema",
                    f"per_model_raw[{model_id!r}][{attr!r}].lo {pm_lo} >= hi {pm_hi}",
                )
            if not (isinstance(pm_rationale, str) and pm_rationale.strip()):
                add(
                    3,
                    "schema",
                    f"per_model_raw[{model_id!r}][{attr!r}].rationale is empty",
                )

    # Invariant 29: failed_models[m] shape + allowlist
    for model_id, failure in failed_dict.items():
        if not isinstance(failure, dict):
            add(3, "type", f"failed_models[{model_id!r}] must be dict")
            continue
        if set(failure.keys()) != _FAILED_MODEL_KEYS:
            add(
                3,
                "schema",
                f"failed_models[{model_id!r}] keys must be {sorted(_FAILED_MODEL_KEYS)}",
            )
            continue
        et = failure["error_type"]
        prov = failure["provider"]
        pr = failure["public_reason"]
        if not (isinstance(et, str) and et):
            add(3, "schema", f"failed_models[{model_id!r}].error_type empty")
        if not (isinstance(prov, str) and prov in ALLOWED_PROVIDERS):
            add(
                3,
                "schema",
                f"failed_models[{model_id!r}].provider {prov!r} not in allowlist",
            )
        if not (isinstance(pr, str) and pr in ALLOWED_PUBLIC_REASONS):
            add(
                3,
                "schema",
                f"failed_models[{model_id!r}].public_reason {pr!r} not in allowlist",
            )

    # Invariant 30: filename matches snapshot["version"]
    if snapshot_path is not None and version_valid and isinstance(version, str):
        stem = snapshot_path.stem
        if stem != version:
            add(
                3,
                "version_filename_mismatch",
                f"snapshot_path stem {stem!r} != snapshot['version'] {version!r}",
            )

    # Invariant 31: version_not_newer (skipped when idempotent_resume_context permits)
    if (
        check_version_not_newer
        and active_snapshot is not None
        and version_valid
        and isinstance(version, str)
    ):
        skip_v31 = False
        if idempotent_resume_context is not None and idempotent_resume_context.get(
            "dest_byte_equal"
        ):
            # Allow equal-version resume only; disallow downgrade.
            active_v = active_snapshot.get("version")
            if isinstance(active_v, str) and version >= active_v:
                skip_v31 = True
        if not skip_v31:
            active_v = active_snapshot.get("version")
            if isinstance(active_v, str) and not (version > active_v):
                add(
                    1,
                    "version_not_newer",
                    f"pending version {version!r} not strictly newer than active {active_v!r}",
                )

    return violations


def _is_list_of_nonempty_strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(s, str) and s for s in value)


def _extract_utc_year_month(generated_at: str) -> str | None:
    """Extract YYYY-MM from a UTC ISO-8601 timestamp.

    Tight validation (architect/code-reviewer HIGH fix):
      - Month: 01-12 only
      - Day: 01-31 only (calendar-day accuracy via datetime.fromisoformat below)
      - Hour: 00-23, Minute/Second: 00-59
      - Offset: only `Z` or `+00:00`. RFC 3339's `-00:00` (unknown local offset)
        is rejected; the plan mandates UTC.
      - Real calendar date validity (e.g., Feb 30 rejected) via datetime parse.
    """
    if not re.fullmatch(
        r"\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"
        r"T([01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?(Z|\+00:00)",
        generated_at,
    ):
        return None
    # Calendar validity check (Feb 30, Apr 31, etc.) via datetime.fromisoformat.
    # Python's fromisoformat accepts '+00:00' natively; convert trailing 'Z'.
    iso = generated_at[:-1] + "+00:00" if generated_at.endswith("Z") else generated_at
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return f"{dt.year:04d}-{dt.month:02d}"


# ---------------------------------------------------------------------------
# Report schema validation
# ---------------------------------------------------------------------------

_REPORT_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "exit_code",
        "violations",
        "pending_path",
        "pending_version",
        "pending_sha256",
        "active_pointer_path",
        "active_pointer_sha256",
        "active_snapshot_path",
        "active_snapshot_sha256",
        "active_version",
    }
)


_REPORT_PATH_KEYS: tuple[str, ...] = (
    "pending_path",
    "active_pointer_path",
    "active_snapshot_path",
)


_REPORT_VERSION_KEYS: tuple[str, ...] = ("pending_version", "active_version")


_REPORT_SHA_KEYS: tuple[str, ...] = (
    "pending_sha256",
    "active_pointer_sha256",
    "active_snapshot_sha256",
)


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


_ACTIVE_CATEGORIES: frozenset[str] = frozenset(
    {
        "missing_active_pointer",
        "active_pointer_malformed",
        "missing_active_snapshot",
        "active_snapshot_malformed",
    }
)


# NOTE: an earlier design used a `_PENDING_MALFORMED_CATEGORIES` frozenset to
# decide pending-shape dispatch by category membership. This was architecturally
# wrong (conflated parse-failure with post-parse semantic invariants — e.g.
# band_too_narrow / disjoint_bands / scenario_envelope_overlap / prompt_hash_mismatch
# all fire AFTER successful parse with a valid pending_version, but the frozenset
# rejected such valid reports). The correct dispatch signal is `pending_version is None`
# (snapshot could not be parsed far enough to derive a version), NOT category names.
# See _validate_report_schema for the pv-driven dispatch.


# Categories that require a fully-readable active snapshot. These compare
# pending against active attribute values, model set, ensemble method, prompt
# hash, or version. Invariant 31 (version_not_newer) requires `active_snapshot`
# to be a parsed dict; downgrade_attempt likewise needs active version derived
# from the parsed body. ANY active-state category (missing/malformed pointer
# OR missing/malformed snapshot body) breaks this requirement.
_REQUIRES_VALID_ACTIVE_SNAPSHOT_CATEGORIES: frozenset[str] = frozenset(
    {
        "magnitude",
        "drift_model_set",
        "drift_ensemble",
        "drift_prompt",
        "version_not_newer",
        "downgrade_attempt",
    }
)

# Categories that require only the active pointer file to EXIST (not be valid).
# `bootstrap_with_existing_active` fires precisely because active.json was
# found on disk; it is compatible with `active_pointer_malformed`,
# `missing_active_snapshot`, and `active_snapshot_malformed` (pointer exists
# in all three) but incompatible with `missing_active_pointer` and all-null.
_REQUIRES_ACTIVE_POINTER_EXISTS_CATEGORIES: frozenset[str] = frozenset(
    {
        "bootstrap_with_existing_active",
    }
)

# Note: range / allowed_range / schema / type / not_finite / etc. are
# pending-side only and never require active context.
# collision_different_content compares pending against DEST (a previously
# written file), not against active, so it is also active-independent.


# Categories that require the pending file to exist and be readable (bytes
# or parsed JSON). These cannot co-occur with `missing_pending_file`.
# Mirrors Round 5 active-side pattern.
_REQUIRES_PENDING_CONTENT_CATEGORIES: frozenset[str] = frozenset(
    {
        # Require parsed pending JSON content
        "range",
        "allowed_range",
        "version_not_newer",
        "schema",
        "unknown_key",
        "type",
        "cross_attr_sanity",
        "derived_metadata_mismatch",
        "prompt_hash_mismatch",
        "version_filename_mismatch",
        "not_finite",
        "generated_at_invalid",
        "prompt_version_invalid",
        "prompt_schema_type_error",
        "scenario_envelope_overlap",
        "disjoint_bands",
        "insufficient_ensemble",
        "band_too_narrow",
        "downgrade_attempt",
        # Active-vs-pending comparison categories (pending must also be parsed)
        "magnitude",
        "drift_model_set",
        "drift_ensemble",
        "drift_prompt",
        # Compare pending bytes against DEST file
        "collision_different_content",
        # Pending must be parsed far enough to know the prompt reference
        "missing_prompt_file",
    }
)

# Parse-failure signal categories: these ARE the signals that describe a
# malformed/unparseable pending snapshot. They CAN fire when validator could
# not derive pending_version, and a canonical "pending corrupt -> swap exit 3"
# report carries them together with pending_version=None (plan validator
# section line 667). Generated_at_invalid and prompt_version_invalid are
# included because they fire from independent field checks that do NOT require
# snapshot["version"] to be valid — see validate_snapshot_invariants
# invariants 7 and 8.
_PARSE_FAILURE_SIGNAL_CATEGORIES: frozenset[str] = frozenset(
    {
        "schema",
        "unknown_key",
        "type",
        "not_finite",
        "generated_at_invalid",
        "prompt_version_invalid",
    }
)

# Categories that require pending_version to be derivable. Subset of
# _REQUIRES_PENDING_CONTENT_CATEGORIES with parse-failure signals excluded.
#
# IMPORTANT: collision_different_content IS included (i.e., NOT excluded) per
# plan's swap procedure: pending.version is validated BEFORE dest path is
# resolved, and collision check happens AFTER. Therefore a collision
# violation implies pending was parsed far enough to have a valid version.
# The defensive sha256-required check in the malformed-pending branch remains
# as belt-and-suspenders but is now logically unreachable.
_REQUIRES_PENDING_VERSION_CATEGORIES: frozenset[str] = (
    _REQUIRES_PENDING_CONTENT_CATEGORIES - _PARSE_FAILURE_SIGNAL_CATEGORIES
)


def _validate_report_schema(report: object) -> None:
    """Verify report dict shape, types, and code/category consistency.

    Accepts `object` (not bare dict) because JSON parsing yields any top-level
    value. The first check is isinstance; on success the function uses
    `typing.cast(Mapping[str, object], report)` for typed access.

    Raises ValueError on any malformed input. Swap script converts to exit 3.

    ## Contract with run_validation (Codex round 8)

    `validate_snapshot_invariants()` may emit post-parse semantic violations
    (e.g., `cross_attr_sanity`, `derived_metadata_mismatch`, `band_too_narrow`)
    even when `snapshot["version"]` failed invariant 6, because each invariant
    runs independently. `run_validation()` is therefore responsible for
    reconciling the violation list with the report shape:

      - If `snapshot["version"]` is regex-valid: report.pending_version is
        set to that value; ALL violations from validate_snapshot_invariants
        are passed through.
      - If `snapshot["version"]` is missing/invalid: report.pending_version
        is set to None, and run_validation MUST emit only categories that
        are coherent with malformed-pending state — i.e., the union of
        `_PARSE_FAILURE_SIGNAL_CATEGORIES` and any active-side violations.
        Post-parse semantic violations (band_too_narrow, derived_metadata_mismatch,
        cross_attr_sanity, scenario_envelope_overlap, etc.) must be suppressed
        because their parse-time prerequisites are absent.

    This guard enforces the second branch above: a malformed-pending report
    carrying a post-parse semantic violation is rejected as inconsistent with
    its declared pending state.
    """
    if not isinstance(report, dict):
        raise ValueError(f"report top-level is {type(report).__name__}, expected dict")

    # NH1.HIGH: explicit cast call after isinstance narrows from `object` to dict
    r: Mapping[str, object] = cast(Mapping[str, object], report)

    # Required keys
    report_keys = set(r.keys())
    missing = _REPORT_REQUIRED_KEYS - report_keys
    if missing:
        raise ValueError(f"report missing required keys: {sorted(missing)}")

    exit_code = r["exit_code"]
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ValueError(f"exit_code must be int, got {type(exit_code).__name__}")
    if exit_code not in ALLOWED_EXIT_CODES:
        raise ValueError(f"exit_code {exit_code} not in {sorted(ALLOWED_EXIT_CODES)}")

    violations = r["violations"]
    if not isinstance(violations, list):
        raise ValueError("violations must be list")

    # exit_code <-> violations consistency
    if exit_code == 0 and violations:
        raise ValueError("exit_code 0 must have empty violations")
    if exit_code != 0 and not violations:
        raise ValueError(f"exit_code {exit_code} must have non-empty violations")

    # Per-violation shape, category allowlist, code-category mapping
    for v in violations:
        if not isinstance(v, dict):
            raise ValueError(f"violation must be dict, got {type(v).__name__}")
        if set(v.keys()) != {"code", "category", "message"}:
            raise ValueError(f"violation keys must be {{code, category, message}}: {v}")
        if not isinstance(v["code"], int) or isinstance(v["code"], bool):
            raise ValueError(f"violation.code must be int: {v}")
        if not isinstance(v["category"], str):
            raise ValueError(f"violation.category must be str: {v}")
        if not isinstance(v["message"], str):
            raise ValueError(f"violation.message must be str: {v}")
        if v["category"] not in CATEGORY_TO_CODE:
            raise ValueError(f"unknown category: {v['category']!r}")
        if v["code"] != CATEGORY_TO_CODE[v["category"]]:
            raise ValueError(
                f"violation code {v['code']} does not match category "
                f"{v['category']!r} (expected {CATEGORY_TO_CODE[v['category']]})"
            )

    # exit_code must equal priority-dispatch result of violations
    if violations:
        present_codes = {int(v["code"]) for v in violations}
        expected_exit = 0
        for c in PRIORITY_ORDER:
            if c in present_codes:
                expected_exit = c
                break
        if exit_code != expected_exit:
            raise ValueError(
                f"exit_code {exit_code} != priority-dispatch {expected_exit}"
            )

    # Path fields: str or None
    for k in _REPORT_PATH_KEYS:
        val = r[k]
        if val is not None and not isinstance(val, str):
            raise ValueError(f"{k} must be str or None, got {type(val).__name__}")

    # Version fields: str or None
    for k in _REPORT_VERSION_KEYS:
        val = r[k]
        if val is not None and not isinstance(val, str):
            raise ValueError(f"{k} must be str or None, got {type(val).__name__}")

    # SHA fields: 64-hex str or None
    for k in _REPORT_SHA_KEYS:
        val = r[k]
        if val is None:
            continue
        if not isinstance(val, str) or not _HEX64.match(val):
            raise ValueError(f"{k} must be 64 lowercase hex chars or None, got {val!r}")

    # BLOCKER (Codex round 3): pending shape dispatch driven by pending_version
    # presence, NOT by category-set membership. Post-parse semantic violations
    # (band_too_narrow, disjoint_bands, insufficient_ensemble, prompt_hash_mismatch,
    # scenario_envelope_overlap, derived_metadata_mismatch, missing_prompt_file,
    # bootstrap_with_existing_active, collision_different_content, downgrade_attempt,
    # etc.) fire AFTER successful parse with a valid pending_version; classifying
    # them as malformed-pending was wrong.
    #
    # Canonical shapes:
    #   missing_pending_file: pending_path non-null; pending_version null; pending_sha256 null
    #   malformed pending (pending_version is None): pending_path non-null;
    #                          pending_sha256 may be non-null if bytes were readable
    #   normal pending (pending_version is non-null): all three non-null;
    #                          pending_version matches _VERSION_PATTERN
    categories_present = {v["category"] for v in violations}
    pp = r["pending_path"]
    pv = r["pending_version"]
    ps = r["pending_sha256"]
    if "missing_pending_file" in categories_present:
        if pp is None:
            raise ValueError("missing_pending_file: pending_path must be non-null")
        if pv is not None:
            raise ValueError("missing_pending_file: pending_version must be null")
        if ps is not None:
            raise ValueError("missing_pending_file: pending_sha256 must be null")
        # HIGH (Codex round 6): pending-content-requiring violations cannot
        # co-occur with missing_pending_file. No bytes were readable, no JSON
        # was parsed — schema invariant checks, semantic violations,
        # collision-vs-dest, and drift comparisons all need pending content.
        blocked_by_missing = (
            categories_present & _REQUIRES_PENDING_CONTENT_CATEGORIES
        )
        if blocked_by_missing:
            raise ValueError(
                f"violations {sorted(blocked_by_missing)} require readable/parsed "
                f"pending content and cannot coexist with missing_pending_file"
            )
    elif pv is None:
        # Malformed pending: pending JSON could not be parsed far enough to
        # extract a version. pending_path is still recorded; pending_sha256 may
        # be non-null if bytes were readable before parse failure (no strict rule).
        if pp is None:
            raise ValueError(
                "malformed-pending (pending_version is None): "
                "pending_path must be non-null"
            )
        # Categories that require a successfully parsed pending_version cannot
        # fire in malformed-pending state (per Codex rounds 7-8 cumulative).
        # The permitted-set is `_PARSE_FAILURE_SIGNAL_CATEGORIES`:
        #   {schema, unknown_key, type, not_finite, generated_at_invalid,
        #    prompt_version_invalid}
        # These six categories describe the malformed state itself and can
        # coexist with pending_version=None.
        #
        # `collision_different_content` is NOT permitted here. Plan's swap
        # procedure validates pending.version BEFORE deriving the dest path
        # and checking for collision; a collision violation therefore implies
        # pending was parsed far enough to have a valid version. It belongs
        # to `_REQUIRES_PENDING_VERSION_CATEGORIES` and is blocked accordingly.
        blocked_by_malformed = (
            categories_present & _REQUIRES_PENDING_VERSION_CATEGORIES
        )
        if blocked_by_malformed:
            raise ValueError(
                f"violations {sorted(blocked_by_malformed)} require a parsed "
                f"pending_version and cannot coexist with malformed-pending state"
            )
        # Defense in depth (Codex round 7): even if collision_different_content
        # were somehow permitted in this branch by a future change, requiring
        # pending_sha256 to be non-null preserves the "bytes-readable" invariant
        # the category logically depends on. As of round 8 this branch is
        # unreachable (collision is blocked by _REQUIRES_PENDING_VERSION_CATEGORIES
        # above) but is retained as belt-and-suspenders.
        if "collision_different_content" in categories_present and ps is None:
            raise ValueError(
                "collision_different_content cannot coexist with malformed-pending "
                "AND pending_sha256=None (bytes were not readable for comparison)"
            )
    else:
        # Normal pending: parsed successfully, version is set. This holds for
        # ALL post-parse violation categories (code 1 drift/magnitude/range/etc.,
        # code 3 semantic mismatches, code 5/6/7 ensemble failures, code 8 swap
        # conditions). Validate version format and require pending_sha256 (bytes
        # were necessarily read to derive a version).
        # pv has been type-checked as str-or-None earlier; narrow for mypy strict.
        assert isinstance(pv, str)
        if not _VERSION_PATTERN.match(pv):
            raise ValueError(
                f"parsed-pending: pending_version {pv!r} does not match _VERSION_PATTERN"
            )
        if pp is None:
            raise ValueError(
                "parsed-pending: pending_path must be non-null when pending_version is set"
            )
        if ps is None:
            raise ValueError(
                "parsed-pending: pending_sha256 must be non-null when pending_version is set"
            )

    # BLOCKER 1: per-category active fields shape validation.
    # Plan canonical shapes:
    #   bootstrap / missing_active_pointer: ALL active_* null
    #   active_pointer_malformed: pointer_path non-null; pointer_sha256 non-null
    #                              if readable; remaining 3 null
    #   missing_active_snapshot: pointer_path/sha256 + active_version +
    #                            active_snapshot_path non-null; active_snapshot_sha256 null
    #   active_snapshot_malformed: ALL active_* non-null
    #   normal-active (exit 0 or only pending/code-1 violations): ALL active_* non-null
    ap_path = r["active_pointer_path"]
    ap_sha = r["active_pointer_sha256"]
    as_path = r["active_snapshot_path"]
    as_sha = r["active_snapshot_sha256"]
    av = r["active_version"]
    # categories_present already computed earlier; reuse it.
    active_cats_present = categories_present & _ACTIVE_CATEGORIES
    if len(active_cats_present) > 1:
        raise ValueError(
            f"at most one active-state category expected, got {sorted(active_cats_present)}"
        )

    # HIGH (Codex round 5): cross-category compatibility — context-requiring
    # categories must be compatible with the active-state. This check fires
    # BEFORE the per-active-state shape branches so it applies regardless of
    # which active-state category (if any) is present.
    snapshot_required = categories_present & _REQUIRES_VALID_ACTIVE_SNAPSHOT_CATEGORIES
    if snapshot_required and active_cats_present:
        raise ValueError(
            f"violations {sorted(snapshot_required)} require a fully-readable "
            f"active snapshot and cannot coexist with active-state category "
            f"{sorted(active_cats_present)}"
        )

    pointer_required = categories_present & _REQUIRES_ACTIVE_POINTER_EXISTS_CATEGORIES
    if pointer_required and "missing_active_pointer" in active_cats_present:
        raise ValueError(
            f"violations {sorted(pointer_required)} fire because an active pointer "
            f"exists and cannot coexist with missing_active_pointer"
        )

    if "active_pointer_malformed" in active_cats_present:
        if ap_path is None:
            raise ValueError("active_pointer_malformed: active_pointer_path must be non-null")
        if av is not None:
            raise ValueError("active_pointer_malformed: active_version must be null")
        if as_path is not None:
            raise ValueError("active_pointer_malformed: active_snapshot_path must be null")
        if as_sha is not None:
            raise ValueError("active_pointer_malformed: active_snapshot_sha256 must be null")
    elif "missing_active_snapshot" in active_cats_present:
        if ap_path is None or ap_sha is None or av is None or as_path is None:
            raise ValueError(
                "missing_active_snapshot: active_pointer_path, active_pointer_sha256, "
                "active_version, and active_snapshot_path must be non-null"
            )
        if as_sha is not None:
            raise ValueError(
                "missing_active_snapshot: active_snapshot_sha256 must be null"
            )
    elif "active_snapshot_malformed" in active_cats_present:
        if any(x is None for x in (ap_path, ap_sha, as_path, as_sha, av)):
            raise ValueError(
                "active_snapshot_malformed: all active_* fields must be non-null"
            )
    elif "missing_active_pointer" in active_cats_present:
        if any(x is not None for x in (ap_path, ap_sha, as_path, as_sha, av)):
            raise ValueError(
                "missing_active_pointer: all active_* fields must be null"
            )
    else:
        # No active-state category: either bootstrap (all null) or normal (all non-null)
        all_none = all(x is None for x in (ap_path, ap_sha, as_path, as_sha, av))
        all_set = all(x is not None for x in (ap_path, ap_sha, as_path, as_sha, av))
        if not (all_none or all_set):
            raise ValueError(
                f"active_* fields must be all-None (bootstrap) or all-non-None "
                f"(normal-active) when no active-state category is present: "
                f"active_pointer_path={ap_path!r}, active_pointer_sha256={ap_sha!r}, "
                f"active_snapshot_path={as_path!r}, active_snapshot_sha256={as_sha!r}, "
                f"active_version={av!r}"
            )
        # Bootstrap path (all_none, no active-state category): no context-
        # requiring categories may co-occur. Both snapshot-required and
        # pointer-required sets are incompatible with all-null active.
        if all_none:
            blocked = categories_present & (
                _REQUIRES_VALID_ACTIVE_SNAPSHOT_CATEGORIES
                | _REQUIRES_ACTIVE_POINTER_EXISTS_CATEGORIES
            )
            if blocked:
                raise ValueError(
                    f"active-context violations {sorted(blocked)} cannot fire "
                    f"with all-null active fields (no active snapshot/pointer)"
                )
