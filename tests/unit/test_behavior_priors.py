"""Tests for src/persona/behavior_priors.py.

Public-core safe: does NOT import PersonaTemplate (which lives in a
private-side module that is denylisted from public-core export). Uses
SimpleNamespace to satisfy the TemplateLike structural protocol.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.persona.behavior_priors import (
    DEFAULT_BOUNDS,
    MAPPED_ATTRS,
    NAS_CALIBRATION_DISCLOSURE,
    PIPELINE_MISSION,
    REACTION_PRIOR_DISCLOSURE,
    SCENARIOS,
    BehaviorPriorSample,
    ScenarioBehaviorPriorProvider,
    TemplateBehaviorPriorProvider,
    _map_to_band,
)


def _fake_template(
    session: float = 0.5,
    search: float = 0.5,
    click: float = 0.2,
    std: float = 0.1,
) -> SimpleNamespace:
    return SimpleNamespace(
        session_length_tendency=session,
        search_propensity=search,
        click_propensity=click,
        session_length_tendency_std=std,
        search_propensity_std=std,
        click_propensity_std=std,
    )


# --- Template provider ---


def test_template_provider_returns_template_means_verbatim():
    rng = np.random.default_rng(0)
    tmpl = _fake_template(session=0.6, search=0.3, click=0.25)
    provider = TemplateBehaviorPriorProvider()
    sample = provider.sample(rng, tmpl)
    assert sample.session_length_tendency == 0.6
    assert sample.search_propensity == 0.3
    assert sample.click_propensity == 0.25


def test_template_provider_std_scale_is_identity():
    provider = TemplateBehaviorPriorProvider()
    for attr in MAPPED_ATTRS:
        assert provider.std_scale(attr) == 1.0


def test_template_provider_std_scale_rejects_unknown_attr():
    provider = TemplateBehaviorPriorProvider()
    with pytest.raises(ValueError, match="Unknown attribute"):
        provider.std_scale("bad_attr")


def test_template_provider_provenance_label():
    provider = TemplateBehaviorPriorProvider()
    assert "explicit simulation assumption" in provider.provenance_label
    assert "PersonaTemplate default" in provider.provenance_label
    assert provider.scenario_name is None


# --- _map_to_band algorithmic tests ---


def test_map_to_band_identity_at_base():
    """When source == target, mapping is identity for any in-range value."""
    band = (0.30, 0.60)
    for v in [0.30, 0.45, 0.60]:
        assert _map_to_band(v, band, band) == pytest.approx(v)


def test_map_to_band_boundary_exactness():
    src = (0.30, 0.60)
    tgt = (0.20, 0.30)
    assert _map_to_band(src[0], src, tgt) == pytest.approx(tgt[0])
    assert _map_to_band(src[1], src, tgt) == pytest.approx(tgt[1])


def test_map_to_band_monotonicity():
    src = (0.10, 0.30)
    tgt = (0.05, 0.50)
    inputs = [0.10, 0.15, 0.20, 0.25, 0.30]
    outputs = [_map_to_band(v, src, tgt) for v in inputs]
    assert outputs == sorted(outputs)


def test_map_to_band_out_of_source_clamps():
    src = (0.10, 0.30)
    tgt = (0.05, 0.15)
    assert _map_to_band(0.05, src, tgt) == pytest.approx(tgt[0])  # below src lo
    assert _map_to_band(0.40, src, tgt) == pytest.approx(tgt[1])  # above src hi


def test_map_to_band_rejects_zero_or_negative_width_source():
    with pytest.raises(ValueError, match="zero or negative width"):
        _map_to_band(0.5, (0.3, 0.3), (0.0, 1.0))
    with pytest.raises(ValueError, match="zero or negative width"):
        _map_to_band(0.5, (0.6, 0.3), (0.0, 1.0))


# --- Scenario provider numeric verification (per-attribute) ---


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("conservative", [0.200, 0.267, 0.300]),
        ("base", [0.300, 0.467, 0.550]),
        ("optimistic", [0.550, 0.683, 0.750]),
    ],
)
def test_scenario_session_length_numeric_table(scenario, expected):
    """Inputs {0.3, 0.5, 0.6} for session_length, source (0.30, 0.60)."""
    provider = ScenarioBehaviorPriorProvider(scenario)
    rng = np.random.default_rng(0)
    actuals = []
    for v in [0.3, 0.5, 0.6]:
        tmpl = _fake_template(session=v, search=0.5, click=0.2)
        actuals.append(provider.sample(rng, tmpl).session_length_tendency)
    for a, e in zip(actuals, expected):
        assert a == pytest.approx(e, abs=0.005)


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("conservative", [0.200, 0.267, 0.300]),
        ("base", [0.300, 0.467, 0.550]),
        ("optimistic", [0.550, 0.683, 0.750]),
    ],
)
def test_scenario_search_propensity_numeric_table(scenario, expected):
    """Inputs {0.3, 0.5, 0.6} for search, source (0.30, 0.60)."""
    provider = ScenarioBehaviorPriorProvider(scenario)
    rng = np.random.default_rng(0)
    actuals = []
    for v in [0.3, 0.5, 0.6]:
        tmpl = _fake_template(session=0.5, search=v, click=0.2)
        actuals.append(provider.sample(rng, tmpl).search_propensity)
    for a, e in zip(actuals, expected):
        assert a == pytest.approx(e, abs=0.005)


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("conservative", [0.050, 0.100, 0.150]),
        ("base", [0.150, 0.225, 0.300]),
        ("optimistic", [0.300, 0.400, 0.500]),
    ],
)
def test_scenario_click_propensity_numeric_table(scenario, expected):
    """Inputs {0.1, 0.2, 0.3} for click, source (0.10, 0.30) — must use in-source values."""
    provider = ScenarioBehaviorPriorProvider(scenario)
    rng = np.random.default_rng(0)
    actuals = []
    for v in [0.1, 0.2, 0.3]:
        tmpl = _fake_template(session=0.5, search=0.5, click=v)
        actuals.append(provider.sample(rng, tmpl).click_propensity)
    for a, e in zip(actuals, expected):
        assert a == pytest.approx(e, abs=0.005)


def test_scenario_click_out_of_source_clamps():
    """click input below src lo -> target lo; above src hi -> target hi."""
    provider = ScenarioBehaviorPriorProvider("conservative")
    rng = np.random.default_rng(0)
    low_tmpl = _fake_template(click=0.05)  # below DEFAULT_BOUNDS click lo (0.10)
    high_tmpl = _fake_template(click=0.40)  # above DEFAULT_BOUNDS click hi (0.30)
    assert provider.sample(rng, low_tmpl).click_propensity == pytest.approx(0.05)
    assert provider.sample(rng, high_tmpl).click_propensity == pytest.approx(0.15)


# --- Scenario ordering invariant ---


def test_scenario_ordering_invariant():
    """conservative.upper <= base.lower <= base.upper <= optimistic.lower per attr."""
    for attr in MAPPED_ATTRS:
        cons = getattr(SCENARIOS["conservative"], f"{attr}_range")
        base = getattr(SCENARIOS["base"], f"{attr}_range")
        opt = getattr(SCENARIOS["optimistic"], f"{attr}_range")
        assert cons[1] <= base[0], f"{attr}: conservative.upper > base.lower"
        assert base[0] <= base[1], f"{attr}: base inverted"
        assert base[1] <= opt[0], f"{attr}: base.upper > optimistic.lower"


# --- std_scale correctness ---


def test_scenario_std_scale_proportional_to_band_width():
    provider = ScenarioBehaviorPriorProvider("conservative")
    # click: band 0.05-0.15 (width 0.10); src 0.10-0.30 (width 0.20); ratio 0.5
    assert provider.std_scale("click_propensity") == pytest.approx(0.5)
    # search: band 0.20-0.30 (width 0.10); src 0.30-0.60 (width 0.30); ratio 1/3
    assert provider.std_scale("search_propensity") == pytest.approx(1 / 3)


def test_scenario_std_scale_rejects_unknown_attr():
    provider = ScenarioBehaviorPriorProvider("base")
    with pytest.raises(ValueError, match="Unknown attribute"):
        provider.std_scale("bad_attr")


# --- Construction-time validation ---


def test_scenario_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown scenario"):
        ScenarioBehaviorPriorProvider("bad_scenario")


def test_scenario_provenance_label_includes_scenario_name():
    for name in ("conservative", "base", "optimistic"):
        provider = ScenarioBehaviorPriorProvider(name)
        assert "explicit simulation assumption" in provider.provenance_label
        assert f"scenario={name}" in provider.provenance_label
        assert provider.scenario_name == name


# --- BehaviorPriorSample shape ---


def test_behavior_prior_sample_is_frozen_dataclass():
    sample = BehaviorPriorSample(
        session_length_tendency=0.5, search_propensity=0.4, click_propensity=0.2
    )
    with pytest.raises(Exception):
        sample.session_length_tendency = 0.9  # type: ignore[misc]


# --- Disclosure constants ---


def test_reaction_prior_disclosure_contains_operative_phrases():
    assert REACTION_PRIOR_DISCLOSURE.strip()
    assert "do not forecast" in REACTION_PRIOR_DISCLOSURE
    assert "Do not present" in REACTION_PRIOR_DISCLOSURE
    assert "performance predictions" in REACTION_PRIOR_DISCLOSURE


def test_nas_disclosure_is_one_sentence_about_bias_correction():
    assert NAS_CALIBRATION_DISCLOSURE.strip()
    assert "LLM-side response bias" in NAS_CALIBRATION_DISCLOSURE
    assert "does not ground reaction estimates" in NAS_CALIBRATION_DISCLOSURE


def test_pipeline_mission_includes_uncalibrated_and_validation_clauses():
    assert PIPELINE_MISSION.strip()
    assert "uncalibrated" in PIPELINE_MISSION
    assert "has not been validated against observed campaign outcomes" in PIPELINE_MISSION


def test_disclosures_are_distinct_strings():
    assert REACTION_PRIOR_DISCLOSURE != NAS_CALIBRATION_DISCLOSURE


# --- DEFAULT_BOUNDS sanity ---


def test_default_bounds_covers_mapped_attrs():
    assert set(DEFAULT_BOUNDS) == set(MAPPED_ATTRS)
    for attr, (lo, hi) in DEFAULT_BOUNDS.items():
        assert lo < hi, f"DEFAULT_BOUNDS[{attr}] inverted or zero-width"
