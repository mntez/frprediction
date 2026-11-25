from datetime import date
import math

import pytest

from ml.baselines.poisson import (
    PoissonBaseline,
    evaluate_predictions,
    poisson_probabilities,
    split_chronologically,
    target_labels,
)
from ml.train_poisson_baseline import validate_probabilities


def dataset_row(match_id, match_date, home_goals, away_goals):
    return {
        "provider_match_id": match_id,
        "match_date": match_date,
        "competition": "PL",
        "home_team": "Home",
        "away_team": "Away",
        "home_goals": home_goals,
        "away_goals": away_goals,
        **target_labels(home_goals, away_goals),
        "home_home_matches_played": 4,
        "home_home_goals_scored_avg": 1.5,
        "home_home_goals_conceded_avg": 1.0,
        "away_away_matches_played": 4,
        "away_away_goals_scored_avg": 1.0,
        "away_away_goals_conceded_avg": 1.5,
    }


def test_chronological_split_is_strict_and_keeps_all_cutoff_date_rows_in_test():
    rows = [
        dataset_row(3, "2026-02-02", 1, 0),
        dataset_row(1, "2026-01-31", 2, 1),
        dataset_row(2, "2026-02-01", 0, 0),
    ]

    training, testing = split_chronologically(rows, date(2026, 2, 1))

    assert [row["provider_match_id"] for row in training] == [1]
    assert [row["provider_match_id"] for row in testing] == [2, 3]
    assert max(row["match_date"] for row in training) < min(row["match_date"] for row in testing)
    assert all(date.fromisoformat(row["match_date"]) < date(2026, 2, 1) for row in training)


def test_poisson_goal_probabilities_are_valid_and_sum_to_one():
    for rate in (0.05, 0.7, 1.8, 3.4, 8.0):
        probabilities = poisson_probabilities(rate)
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-12)
        assert probabilities[0] == pytest.approx(math.exp(-rate))
        assert probabilities[1] == pytest.approx(rate * probabilities[0])


def test_prediction_probabilities_are_valid_and_1x2_sums_to_one():
    model = PoissonBaseline.fit([
        dataset_row(1, "2026-01-01", 2, 1),
        dataset_row(2, "2026-01-02", 1, 0),
    ])
    prediction = model.predict(dataset_row(3, "2026-02-01", 0, 0))

    assert sum(prediction[key] for key in ("home_win_probability", "draw_probability", "away_win_probability")) == pytest.approx(1.0)
    assert all(0 <= prediction[key] <= 1 for key in (
        "home_win_probability", "draw_probability", "away_win_probability",
        "over_1_5_probability", "over_2_5_probability", "over_3_5_probability", "btts_probability",
    ))
    assert prediction["predicted_home_score"] >= 0
    assert prediction["predicted_away_score"] >= 0
    validate_probabilities([prediction | {"provider_match_id": 3}])


def test_targets_are_generated_from_actual_score_correctly():
    assert target_labels(2, 1) == {
        "home_goals": 2, "away_goals": 1, "home_win": 1, "draw": 0, "away_win": 0,
        "total_goals": 3, "over_1_5": 1, "over_2_5": 1, "over_3_5": 0, "btts": 1,
    }
    assert target_labels(0, 0)["draw"] == 1
    assert target_labels(0, 0)["btts"] == 0


def test_training_and_evaluation_are_deterministic_and_future_rows_do_not_change_fit():
    rows = [
        dataset_row(1, "2026-01-01", 2, 0),
        dataset_row(2, "2026-01-02", 1, 1),
        dataset_row(3, "2026-02-01", 1, 0),
        dataset_row(4, "2026-02-02", 0, 3),
    ]
    training, testing = split_chronologically(rows, date(2026, 2, 1))
    model = PoissonBaseline.fit(training)
    other_model = PoissonBaseline.fit(training)
    training_with_future, _ = split_chronologically(
        rows + [dataset_row(5, "2026-03-01", 10, 10)], date(2026, 2, 1)
    )

    assert model == other_model
    assert model.home_goal_mean == other_model.home_goal_mean
    assert [row["provider_match_id"] for row in training_with_future] == [1, 2]
    assert PoissonBaseline.fit(training_with_future) == model
    metrics_a, predictions_a = evaluate_predictions(testing, model)
    metrics_b, predictions_b = evaluate_predictions(testing, other_model)
    assert metrics_a == metrics_b
    assert predictions_a == predictions_b

def test_evaluation_rejects_targets_that_do_not_match_actual_scores():
    training = [dataset_row(1, "2026-01-01", 2, 1)]
    model = PoissonBaseline.fit(training)
    bad_test_row = dict(dataset_row(2, "2026-02-01", 1, 0), home_win=0, draw=1)
    with pytest.raises(ValueError, match="does not match scores"):
        evaluate_predictions([bad_test_row], model)
