from datetime import date
import math

import pytest

from ml.baselines.logistic_regression import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FittedPreprocessor,
    evaluate_classifier,
    fit_classifier,
    split_using_poisson_policy,
)
from ml.baselines.poisson import split_chronologically


def row(match_id, match_date, result, home_team="Home A", away_team="Away A"):
    outcome = {"home_win": 0, "draw": 0, "away_win": 0}
    outcome[result] = 1
    values = {column: 1.0 for column in NUMERIC_FEATURES}
    values.update({
        "season_start_year": 2025,
        "home_home_matches_played": 2,
        "home_home_goals_scored_avg": 1.5,
        "home_home_goals_conceded_avg": 1.0,
        "away_away_matches_played": 2,
        "away_away_goals_scored_avg": 1.0,
        "away_away_goals_conceded_avg": 1.5,
    })
    return {
        "provider_match_id": match_id,
        "match_date": match_date,
        "competition": "PL",
        "home_team": home_team,
        "away_team": away_team,
        "home_recent_form": "WWD",
        "away_recent_form": "DLL",
        **values,
        **outcome,
    }


def test_logistic_split_matches_poisson_chronological_split():
    rows = [
        row(3, "2026-02-02", "away_win"),
        row(1, "2026-01-31", "home_win"),
        row(2, "2026-02-01", "draw"),
    ]
    cutoff = date(2026, 2, 1)
    assert split_using_poisson_policy(rows, cutoff) == split_chronologically(rows, cutoff)


def test_preprocessor_fits_imputation_scaling_and_categories_on_training_only():
    training = [row(1, "2026-01-01", "home_win"), row(2, "2026-01-02", "draw")]
    training[0]["home_home_goals_scored_avg"] = 1.0
    training[1]["home_home_goals_scored_avg"] = 3.0
    test_row = row(3, "2026-02-01", "away_win", home_team="Unseen Club")
    test_row["home_home_goals_scored_avg"] = 1000.0
    test_row["away_recent_form"] = None

    preprocessor = FittedPreprocessor.fit(training)
    original_stats = preprocessor.to_dict()
    transformed_test = preprocessor.transform([test_row])[0]

    assert preprocessor.numeric_stats["home_home_goals_scored_avg"] == {
        "median": 2.0, "mean": 2.0, "scale": 1.0,
    }
    assert "Unseen Club" not in preprocessor.categorical_values["home_team"]
    assert preprocessor.to_dict() == original_stats
    assert math.isclose(transformed_test[NUMERIC_FEATURES.index("home_home_goals_scored_avg")], 998.0)
    assert len(transformed_test) == len(preprocessor.feature_names)
    assert all(column in CATEGORICAL_FEATURES for column in preprocessor.categorical_values)


def test_logistic_probabilities_are_finite_and_sum_to_one():
    training = [
        row(1, "2026-01-01", "home_win"),
        row(2, "2026-01-02", "draw"),
        row(3, "2026-01-03", "away_win"),
        row(4, "2026-01-04", "home_win", home_team="Home B"),
        row(5, "2026-01-05", "draw", away_team="Away B"),
        row(6, "2026-01-06", "away_win", home_team="Home C"),
    ]
    preprocessor, classifier = fit_classifier(training)
    metrics, predictions = evaluate_classifier([row(7, "2026-02-01", "home_win")], preprocessor, classifier)

    for prediction in predictions:
        probabilities = [prediction[key] for key in (
            "home_win_probability", "draw_probability", "away_win_probability",
        )]
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-10)
    assert set(metrics["per_class"]) == {"home_win", "draw", "away_win"}
    assert len(metrics["confusion_matrix"]["rows_actual_columns_predicted"]) == 3


def test_test_rows_do_not_change_preprocessing_or_fitted_classifier():
    training = [
        row(1, "2026-01-01", "home_win"), row(2, "2026-01-02", "draw"),
        row(3, "2026-01-03", "away_win"), row(4, "2026-01-04", "home_win"),
        row(5, "2026-01-05", "draw"), row(6, "2026-01-06", "away_win"),
    ]
    test = [row(7, "2026-02-01", "home_win", home_team="Future Team")]
    preprocessor_a, model_a = fit_classifier(training)
    preprocessor_b, model_b = fit_classifier(training)
    before = preprocessor_a.to_dict()
    preprocessor_a.transform(test)

    assert preprocessor_a.to_dict() == before
    assert preprocessor_a.to_dict() == preprocessor_b.to_dict()
    assert model_a.classes_.tolist() == model_b.classes_.tolist()
    assert model_a.coef_.tolist() == model_b.coef_.tolist()
    assert model_a.intercept_.tolist() == model_b.intercept_.tolist()


def test_missing_numeric_values_are_median_imputed_with_a_missing_indicator():
    training = [row(1, "2026-01-01", "home_win"), row(2, "2026-01-02", "draw")]
    training[0]["h2h_previous_matches"] = None
    training[1]["h2h_previous_matches"] = 4
    preprocessor = FittedPreprocessor.fit(training)
    transformed = preprocessor.transform([training[0]])[0]
    missing_index = preprocessor.feature_names.index("h2h_previous_matches__missing")

    assert preprocessor.numeric_stats["h2h_previous_matches"]["median"] == 4
    assert transformed[missing_index] == 1.0
