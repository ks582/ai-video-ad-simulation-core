"""Audience targeting criteria for demographic and affinity-based filtering.

Provides ``TargetingCriteria`` — a declarative specification of which
demographics and affinities the ad campaign targets.  The population
generator uses these criteria to apply rejection sampling so that
generated personas match the target audience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.persona.models import (
    AgeGroup,
    EducationLevel,
    GenderEnum,
    IncomeBracket,
    RegionEnum,
    SegmentAttrs,
)


@dataclass(frozen=True)
class DemographicFilter:
    """Inclusive filter — a persona passes if it matches *all* non-empty fields."""

    age_groups: frozenset[AgeGroup] = frozenset()
    genders: frozenset[GenderEnum] = frozenset()
    income_brackets: frozenset[IncomeBracket] = frozenset()
    education_levels: frozenset[EducationLevel] = frozenset()
    regions: frozenset[RegionEnum] = frozenset()

    def matches(self, attrs: SegmentAttrs) -> bool:
        """Return True if *attrs* satisfies every non-empty constraint."""
        if self.age_groups and attrs.age_group not in self.age_groups:
            return False
        if self.genders and attrs.gender not in self.genders:
            return False
        if self.income_brackets and attrs.income_bracket not in self.income_brackets:
            return False
        if self.education_levels and attrs.education_level not in self.education_levels:
            return False
        if self.regions and attrs.region not in self.regions:
            return False
        return True

    @property
    def is_empty(self) -> bool:
        return not (
            self.age_groups
            or self.genders
            or self.income_brackets
            or self.education_levels
            or self.regions
        )


@dataclass(frozen=True)
class TargetingCriteria:
    """Full audience targeting specification for a campaign."""

    demographics: DemographicFilter = field(default_factory=DemographicFilter)
    affinity_ids: tuple[str, ...] = ()
    description: str = ""

    @property
    def is_empty(self) -> bool:
        return self.demographics.is_empty and not self.affinity_ids and not self.description

    def summary(self) -> str:
        """Human-readable one-liner for LLM prompt injection."""
        parts: list[str] = []
        d = self.demographics
        if d.age_groups:
            parts.append(f"Age: {', '.join(sorted(a.value for a in d.age_groups))}")
        if d.genders:
            parts.append(f"Gender: {', '.join(sorted(g.value for g in d.genders))}")
        if d.income_brackets:
            parts.append(f"Income: {', '.join(sorted(i.value for i in d.income_brackets))}")
        if d.education_levels:
            parts.append(f"Education: {', '.join(sorted(e.value for e in d.education_levels))}")
        if d.regions:
            parts.append(f"Region: {', '.join(sorted(r.value for r in d.regions))}")
        if self.affinity_ids:
            parts.append(f"Affinity: {', '.join(self.affinity_ids)}")
        if self.description:
            parts.append(f"Description: {self.description}")
        return "; ".join(parts) if parts else "No targeting (broad reach)"
