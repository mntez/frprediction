import pytest

from ml.analysis.phase_3e import (
    FEATURE_GROUPS,
    _apply_temperature,
    _cluster_bootstrap_delta,
    _fit_temperature,
    build_ablation_plan,
)
from ml.features.feature_engineering_v2 import FEATURE_COLUMNS


def test_ablation_plan_is_finite_and_covers_groups_and_individual_removals():
    plan = build_ablation_plan()
    assert len(plan) == 1 + len(FEATURE_GROUPS) + len(FEATURE_COLUMNS)
    assert plan["full_13"] == tuple(FEATURE_COLUMNS)
    for group, members in FEATURE_GROUPS.items():
        assert set(plan[f"drop_group_{group}"]) == set(FEATURE_COLUMNS) - set(members)
    for feature in FEATURE_COLUMNS:
        assert set(plan[f"drop_feature_{feature}"]) == set(FEATURE_COLUMNS) - {feature}


def _prediction(index):
    return {
        "provider_match_id": index,
        "match_date": f"2026-01-{index:02d}",
        "home_win_probability": 0.5,
        "draw_probability": 0.3,
        "away_win_probability": 0.2,
        "actual_home_win": int(index % 3 == 0),
        "actual_draw": int(index % 3 == 1),
        "actual_away_win": int(index % 3 == 2),
        "expected_home_goals": 1.5,
        "expected_away_goals": 1.1,
        "actual_home_goals": 1,
        "actual_away_goals": 0,
        "actual_total_goals": 1,
    }


def test_temperature_is_fit_from_train_only_and_preserves_goal_rates():
    train = [_prediction(i) for i in range(1, 25)]
    validation = [_prediction(i) for i in range(25, 29)]
    fitted = _fit_temperature(train)
    calibrated = _apply_temperature(validation, fitted["temperature"])
    assert fitted["fit_rows"] == len(train)
    assert "training partition only" in fitted["fit_data"]
    assert all(abs(sum(row[key] for key in ("home_win_probability", "draw_probability", "away_win_probability")) - 1) < 1e-12 for row in calibrated)
    assert [(row["expected_home_goals"], row["expected_away_goals"]) for row in calibrated] == [(row["expected_home_goals"], row["expected_away_goals"]) for row in validation]


def test_week_cluster_paired_bootstrap_is_deterministic_and_requires_aligned_ids():
    a = [_prediction(i) for i in range(1, 25)]
    b = [dict(row, home_win_probability=0.48, draw_probability=0.31, away_win_probability=0.21) for row in a]
    first = _cluster_bootstrap_delta(a, b, "brier", seed=5, replicates=100)
    second = _cluster_bootstrap_delta(a, b, "brier", seed=5, replicates=100)
    assert first == second
    assert first["paired_fixture_count"] == 24
    assert len(first["ci_95_week_block_percentile"]) == 2
    with pytest.raises(ValueError, match="identical ordered fixture IDs"):
        _cluster_bootstrap_delta(a, b[:-1], "brier", replicates=10)

