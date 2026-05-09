"""T-502: Pair count and statistical configuration utility.

Provides pair count configuration, power calculation assistance, and recommended settings per primary metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from src.config_loader import get_config


class PowerRecommendation(BaseModel):
    """Recommended settings per metric."""

    metric_name: str
    metric_type: str = "continuous"  # "continuous" or "binary"
    min_pairs: int = 100
    recommended_pairs: int = 200
    min_seeds: int = 3
    recommended_seeds: int = 10
    rationale: str = ""


class PowerEstimate(BaseModel):
    """Statistical power estimation result."""

    n_pairs: int
    effect_size: float
    alpha: float = 0.05
    power: float = 0.0
    sufficient: bool = False


class PowerPlanner:
    """Pair count and statistical configuration utility.

    Statistical requirements:
    - MUST: Assume repeated execution with multiple seeds
    - MUST: Pre-estimate the required pair count for each primary metric
    - SHOULD: Use 100-200 pairs as the minimum starting point for continuous metrics
    - SHOULD: Assume 500-2,000+ pairs for binary metrics
    - SHOULD: Prefer paired design
    """

    @staticmethod
    def _build_default_recommendations() -> dict[str, dict]:
        """Build default recommendations from config."""
        cfg_pw = get_config()["power"]
        base = {
            "metric_type": "continuous",
            "min_pairs": cfg_pw["default_min_pairs"],
            "recommended_pairs": cfg_pw["default_recommended_pairs"],
            "min_seeds": cfg_pw["default_min_seeds"],
            "recommended_seeds": cfg_pw["default_recommended_seeds"],
        }
        metrics = {
            "brand_awareness_lift": "brand_recall difference",
            "ad_recall_lift": "ad_recall difference",
            "message_association_lift": "message_association difference",
            "purchase_intent_lift": "purchase_intent difference",
            "search_intent_lift": "search probability difference",
            "favorability_lift": "favorability difference, 0-1 scale",
            "consideration_lift": "consideration difference",
        }
        return {
            name: {**base, "rationale": f"Continuous metric ({desc}). {base['min_pairs']}-{base['recommended_pairs']} pairs as minimum guideline."}
            for name, desc in metrics.items()
        }

    def get_recommendations(self) -> list[PowerRecommendation]:
        """Return recommended settings for each primary metric."""
        recs = []
        for name, cfg in self._build_default_recommendations().items():
            recs.append(PowerRecommendation(metric_name=name, **cfg))
        return recs

    def estimate_power(
        self,
        n_pairs: int,
        effect_size: float,
        alpha: float | None = None,
        metric_type: str = "continuous",
    ) -> PowerEstimate:
        """Approximate power estimation (based on paired t-test).

        Power calculation using normal approximation:
        power ~ Phi(|d|*sqrt(n) - z_{alpha/2})

        Args:
            n_pairs: Number of pairs
            effect_size: Cohen's d (standardized effect size)
            alpha: Significance level
            metric_type: "continuous" or "binary"
        """
        from scipy.stats import norm

        cfg_pw = get_config()["power"]
        if alpha is None:
            alpha = cfg_pw["default_alpha"]
        z_alpha = norm.ppf(1 - alpha / 2)
        noncentrality = abs(effect_size) * math.sqrt(n_pairs)
        power = 1 - norm.cdf(z_alpha - noncentrality)
        power = max(0.0, min(1.0, power))

        return PowerEstimate(
            n_pairs=n_pairs,
            effect_size=effect_size,
            alpha=alpha,
            power=round(power, 4),
            sufficient=power >= cfg_pw["target_power"],
        )

    def required_pairs(
        self,
        effect_size: float,
        alpha: float | None = None,
        target_power: float | None = None,
    ) -> int:
        """Calculate the number of pairs required to achieve the target power.

        n >= ((z_{alpha/2} + z_{beta}) / d)^2

        Args:
            effect_size: Cohen's d
            alpha: Significance level
            target_power: Target statistical power
        """
        from scipy.stats import norm

        cfg_pw = get_config()["power"]
        if alpha is None:
            alpha = cfg_pw["default_alpha"]
        if target_power is None:
            target_power = cfg_pw["target_power"]
        if effect_size <= 0:
            return cfg_pw["max_pairs_guard"]

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(target_power)
        n = ((z_alpha + z_beta) / effect_size) ** 2
        return int(math.ceil(n))
