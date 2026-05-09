"""T-502 unit tests: PowerPlanner."""

from src.analysis.power_planner import PowerPlanner, PowerRecommendation


class TestPowerPlanner:
    def test_get_recommendations(self):
        planner = PowerPlanner()
        recs = planner.get_recommendations()
        assert len(recs) == 7
        names = {r.metric_name for r in recs}
        assert "brand_awareness_lift" in names
        assert "ad_recall_lift" in names
        assert "message_association_lift" in names
        assert "purchase_intent_lift" in names
        assert "search_intent_lift" in names
        assert "favorability_lift" in names
        assert "consideration_lift" in names

    def test_binary_metrics_need_more_pairs(self):
        planner = PowerPlanner()
        recs = planner.get_recommendations()
        for r in recs:
            if r.metric_type == "binary":
                assert r.min_pairs >= 500
            elif r.metric_type == "continuous":
                assert r.min_pairs >= 100

    def test_estimate_power(self):
        planner = PowerPlanner()
        result = planner.estimate_power(n_pairs=200, effect_size=0.3)
        assert 0 < result.power <= 1.0
        assert result.n_pairs == 200

    def test_large_n_high_power(self):
        planner = PowerPlanner()
        result = planner.estimate_power(n_pairs=1000, effect_size=0.5)
        assert result.power > 0.95
        assert result.sufficient is True

    def test_small_n_low_power(self):
        planner = PowerPlanner()
        result = planner.estimate_power(n_pairs=10, effect_size=0.1)
        assert result.power < 0.5
        assert result.sufficient is False

    def test_required_pairs(self):
        planner = PowerPlanner()
        n = planner.required_pairs(effect_size=0.3)
        assert n > 0
        # 80% power at d=0.3 should need ~88 pairs
        assert 50 < n < 200


