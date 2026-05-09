"""Joint Demographic Distribution — ancestral sampler using ACS 2023 CPTs.

Implements a conditional probability table (CPT) approach that captures
real-world inter-attribute correlations observed in U.S. Census Bureau
ACS 2023 data.  Instead of sampling all attributes independently from
their marginals, it uses ancestral sampling:

    age → income | age
        → education | age
        → employment | age
    gender  (marginal, independent of age in this model)
    region  (marginal, independent of age in this model)

CPT data sources
----------------
- ACS 2023 Table B19001 (household income by age of householder)
- ACS 2023 Table S2301 (employment status by age and sex)
- ACS 2023 Table B15001 (educational attainment by sex and age)
- NTIA Internet Use Survey 2023 for marginal gender/region weights

Notes
-----
All CPT values are approximations derived from publicly available ACS
summary statistics.  They are not re-derived at runtime; they are
hardcoded here to avoid requiring a large ACS PUMS microdata download.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
CPT = dict[str, dict[str, float]]  # {conditioning_value: {outcome: prob}}


# ---------------------------------------------------------------------------
# ACS 2023 conditional probability tables (approximated)
# ---------------------------------------------------------------------------

# P(income | age_group)
# Source: ACS 2023 Table B19001 / S1903, collapsed to our 6 income brackets
_INCOME_GIVEN_AGE: CPT = {
    "18-24": {
        "<25k":     0.52,
        "25k-50k":  0.28,
        "50k-75k":  0.11,
        "75k-100k": 0.05,
        "100k-150k": 0.03,
        "150k+":    0.01,
    },
    "25-34": {
        "<25k":     0.25,
        "25k-50k":  0.28,
        "50k-75k":  0.20,
        "75k-100k": 0.13,
        "100k-150k": 0.10,
        "150k+":    0.04,
    },
    "35-44": {
        "<25k":     0.18,
        "25k-50k":  0.23,
        "50k-75k":  0.20,
        "75k-100k": 0.15,
        "100k-150k": 0.15,
        "150k+":    0.09,
    },
    "45-54": {
        "<25k":     0.17,
        "25k-50k":  0.22,
        "50k-75k":  0.20,
        "75k-100k": 0.16,
        "100k-150k": 0.15,
        "150k+":    0.10,
    },
    "55-64": {
        "<25k":     0.22,
        "25k-50k":  0.25,
        "50k-75k":  0.20,
        "75k-100k": 0.14,
        "100k-150k": 0.12,
        "150k+":    0.07,
    },
    "65+": {
        "<25k":     0.30,
        "25k-50k":  0.30,
        "50k-75k":  0.18,
        "75k-100k": 0.10,
        "100k-150k": 0.08,
        "150k+":    0.04,
    },
}

# P(education | age_group)
# Source: ACS 2023 Table B15001 / S1501
_EDUCATION_GIVEN_AGE: CPT = {
    "18-24": {
        "less_than_hs": 0.10,
        "hs_diploma":   0.30,
        "some_college": 0.40,
        "bachelors":    0.15,
        "graduate":     0.05,
    },
    "25-34": {
        "less_than_hs": 0.09,
        "hs_diploma":   0.22,
        "some_college": 0.20,
        "bachelors":    0.33,
        "graduate":     0.16,
    },
    "35-44": {
        "less_than_hs": 0.09,
        "hs_diploma":   0.22,
        "some_college": 0.19,
        "bachelors":    0.32,
        "graduate":     0.18,
    },
    "45-54": {
        "less_than_hs": 0.09,
        "hs_diploma":   0.26,
        "some_college": 0.20,
        "bachelors":    0.29,
        "graduate":     0.16,
    },
    "55-64": {
        "less_than_hs": 0.10,
        "hs_diploma":   0.29,
        "some_college": 0.21,
        "bachelors":    0.26,
        "graduate":     0.14,
    },
    "65+": {
        "less_than_hs": 0.14,
        "hs_diploma":   0.32,
        "some_college": 0.21,
        "bachelors":    0.22,
        "graduate":     0.11,
    },
}

# P(employment_status | age_group)
# Source: ACS 2023 Table S2301
_EMPLOYMENT_GIVEN_AGE: CPT = {
    "18-24": {
        "employed_full_time":  0.30,
        "employed_part_time":  0.25,
        "unemployed":          0.10,
        "not_in_labor_force":  0.15,
        "student":             0.20,
    },
    "25-34": {
        "employed_full_time":  0.65,
        "employed_part_time":  0.12,
        "unemployed":          0.06,
        "not_in_labor_force":  0.12,
        "student":             0.05,
    },
    "35-44": {
        "employed_full_time":  0.70,
        "employed_part_time":  0.12,
        "unemployed":          0.05,
        "not_in_labor_force":  0.12,
        "student":             0.01,
    },
    "45-54": {
        "employed_full_time":  0.69,
        "employed_part_time":  0.12,
        "unemployed":          0.04,
        "not_in_labor_force":  0.14,
        "student":             0.01,
    },
    "55-64": {
        "employed_full_time":  0.54,
        "employed_part_time":  0.14,
        "unemployed":          0.04,
        "not_in_labor_force":  0.27,
        "student":             0.01,
    },
    "65+": {
        "employed_full_time":  0.13,
        "employed_part_time":  0.12,
        "unemployed":          0.02,
        "not_in_labor_force":  0.72,
        "student":             0.01,
    },
}


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class JointDemographicDistribution:
    """Ancestral sampler for correlated demographic attributes.

    Sampling order:
      1. age_group       — from marginal P(age)
      2. income_bracket  — from CPT P(income | age)
      3. education_level — from CPT P(education | age)
      4. employment_status — from CPT P(employment | age)
      5. gender          — from marginal P(gender) [independent]
      6. region          — from marginal P(region) [independent]

    Parameters
    ----------
    marginal_age : dict[str, float]
        Marginal age_group weights (from DemographicDistribution).
    marginal_gender : dict[str, float]
        Marginal gender weights.
    marginal_region : dict[str, float]
        Marginal region weights.
    income_given_age : CPT, optional
        Override for P(income | age). Defaults to ACS 2023 approximation.
    education_given_age : CPT, optional
        Override for P(education | age). Defaults to ACS 2023 approximation.
    employment_given_age : CPT, optional
        Override for P(employment | age). Defaults to ACS 2023 approximation.
    """

    _ACS_DEFAULT_AGE: dict[str, float] = {
        "18-24": 0.128, "25-34": 0.180, "35-44": 0.175,
        "45-54": 0.166, "55-64": 0.171, "65+": 0.180,
    }
    _ACS_DEFAULT_GENDER: dict[str, float] = {
        "male": 0.487, "female": 0.503, "non_binary": 0.007, "unspecified": 0.003,
    }
    _ACS_DEFAULT_REGION: dict[str, float] = {
        "northeast": 0.172, "midwest": 0.209, "south": 0.382, "west": 0.237,
    }

    def __init__(
        self,
        marginal_age: dict[str, float],
        marginal_gender: dict[str, float],
        marginal_region: dict[str, float],
        income_given_age: CPT | None = None,
        education_given_age: CPT | None = None,
        employment_given_age: CPT | None = None,
    ) -> None:
        self._age = marginal_age
        self._gender = marginal_gender
        self._region = marginal_region
        self._income_cpt = income_given_age or _INCOME_GIVEN_AGE
        self._edu_cpt = education_given_age or _EDUCATION_GIVEN_AGE
        self._emp_cpt = employment_given_age or _EMPLOYMENT_GIVEN_AGE

    @classmethod
    def from_acs_defaults(cls) -> "JointDemographicDistribution":
        """Construct using built-in ACS 2023 / NTIA 2023 marginals."""
        return cls(
            marginal_age=cls._ACS_DEFAULT_AGE,
            marginal_gender=cls._ACS_DEFAULT_GENDER,
            marginal_region=cls._ACS_DEFAULT_REGION,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_weights(
        weights: dict[str, float], allowed: set[str] | None,
    ) -> dict[str, float]:
        """Return *weights* filtered to *allowed* keys. Raises if result is empty."""
        if allowed is None:
            return weights
        masked = {k: v for k, v in weights.items() if k in allowed}
        if not masked:
            raise ValueError(
                f"Constraint leaves no valid values. "
                f"Allowed={allowed}, available={set(weights)}"
            )
        return masked

    @staticmethod
    def _sample_from(weights: dict[str, float], rng: np.random.Generator) -> str:
        """Categorical draw from a {category: weight} dict (weights need not sum to 1)."""
        cats = list(weights.keys())
        vals = np.array(list(weights.values()), dtype=float)
        vals /= vals.sum()
        return str(rng.choice(cats, p=vals))

    def _sample_conditional(
        self, cpt: CPT, conditioning_value: str, rng: np.random.Generator
    ) -> str:
        """Sample from P(outcome | conditioning_value) using the given CPT."""
        if conditioning_value not in cpt:
            # Fallback: uniform over known outcomes of first CPT entry
            first_row = next(iter(cpt.values()))
            return self._sample_from({k: 1.0 for k in first_row}, rng)
        return self._sample_from(cpt[conditioning_value], rng)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        rng: np.random.Generator,
        constraints: dict[str, set[str]] | None = None,
    ) -> dict[str, str]:
        """Draw one correlated demographic attribute set.

        Args:
            rng: Random number generator.
            constraints: Optional mapping of attribute name to allowed values.
                Keys are ``age_group``, ``gender``, ``region``,
                ``income_bracket``, ``education_level``, ``employment_status``.
                Only values present in the set are considered during sampling.
                The ancestral correlation structure is preserved within the
                constrained subspace.

        Returns a dict with keys:
            age_group, income_bracket, education_level,
            employment_status, gender, region

        Raises:
            ValueError: If a constraint leaves no valid values for an attribute.
        """
        c = constraints or {}

        age_weights = self._mask_weights(self._age, c.get("age_group"))
        age = self._sample_from(age_weights, rng)

        def _constrained_cpt_sample(
            cpt: CPT, age_key: str, constraint_key: str | None,
        ) -> str:
            row = cpt.get(age_key, next(iter(cpt.values())))
            masked = self._mask_weights(row, c.get(constraint_key) if constraint_key else None)
            return self._sample_from(masked, rng)

        income = _constrained_cpt_sample(self._income_cpt, age, "income_bracket")
        education = _constrained_cpt_sample(self._edu_cpt, age, "education_level")
        employment = _constrained_cpt_sample(self._emp_cpt, age, "employment_status")

        gender_weights = self._mask_weights(self._gender, c.get("gender"))
        gender = self._sample_from(gender_weights, rng)

        region_weights = self._mask_weights(self._region, c.get("region"))
        region = self._sample_from(region_weights, rng)

        return {
            "age_group": age,
            "income_bracket": income,
            "education_level": education,
            "employment_status": employment,
            "gender": gender,
            "region": region,
        }


def _calibrate_cpt(
    cpt: CPT,
    marginal_age: dict[str, float],
    target_marginal: dict[str, float],
) -> CPT:
    """Calibrate a CPT so its implied marginal matches *target_marginal*.

    Uses iterative proportional scaling:
    For each outcome o, scale P(o|a) across all age groups a by
    s_o = target_marginal[o] / implied_marginal[o], then renormalize rows.

    This preserves the relative correlation structure within each age cohort
    while ensuring the population-level marginal matches the NTIA/ACS target.

    Parameters
    ----------
    cpt : CPT
        P(outcome | age_group) table.
    marginal_age : dict[str, float]
        P(age_group) weights (need not sum to exactly 1; will be normalized).
    target_marginal : dict[str, float]
        Desired marginal P(outcome) (NTIA/ACS target).
    """
    age_values = list(marginal_age.keys())
    age_probs = np.array([marginal_age[a] for a in age_values], dtype=float)
    age_probs /= age_probs.sum()

    outcomes = sorted(target_marginal.keys())

    # Run 100 iterations of marginal calibration (IPF-style)
    # Start from the original CPT
    calibrated: dict[str, dict[str, float]] = {
        a: dict(cpt.get(a, {})) for a in age_values
    }

    for _iter in range(100):
        # Compute current implied marginal
        implied: dict[str, float] = {o: 0.0 for o in outcomes}
        for a, prob_a in zip(age_values, age_probs):
            row = calibrated.get(a, {})
            row_sum = sum(row.get(o, 0.0) for o in outcomes) or 1.0
            for o in outcomes:
                implied[o] += (row.get(o, 0.0) / row_sum) * prob_a

        # Scale factors
        scale: dict[str, float] = {}
        for o in outcomes:
            imp = implied.get(o, 0.0)
            tgt = target_marginal.get(o, 0.0)
            scale[o] = tgt / imp if imp > 1e-12 else 1.0

        # Apply scale and renormalize each row
        for a in age_values:
            row = calibrated.get(a, {})
            scaled_row = {o: row.get(o, 0.0) * scale.get(o, 1.0) for o in outcomes}
            row_sum = sum(scaled_row.values()) or 1.0
            calibrated[a] = {o: v / row_sum for o, v in scaled_row.items()}

    return calibrated
