import csv
from datetime import date
import math

import pytest

from ml.baselines.logistic_regression import (
    NUMERIC_FEATURES,
    FittedPreprocessor,
    evaluate_classifier,
)
from ml.baselines.xgboost_classifier import fit_xgboost
from ml.train_xgboost import verify_test_match_ids
from ml.baselines.logistic_regression import split_using_poisson_policy


def make_row(match_id, match_date, label, home_team="Home A", away_team="Away A"):
    classes = {"home_win": 0, "draw": 0, "away_win": 0}
    classes[label] = 1
    features = {column: float((match_id % 5) + 1) for column in NUMERIC_FEATURES}
    features.update({
        "season_start_year": 2025,
        "home_home_matches_played": 3,
        "home_home_goals_scored_avg": 1.5,
        "home_home_goals_conceded_avg": 1.0,
        "away_away_matches_played": 3,
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
        **features,
        **classes,
    }


def sample_training_rows():
    outcomes = ("home_win", "draw", "away_win")
    return [
        make_row(index + 1, f"2026-01-{index + 1:02d}", outcomes[index % 3],
                 home_team=f"Home {index % 4}", away_team=f"Away {index % 5}")
        for index in range(15)
    ]


def test_xgboost_split_matches_existing_chronological_split():
    rows = [
        make_row(3, "2026-02-02", "away_win"),
        make_row(1, "2026-01-31", "home_win"),
        make_row(2, "2026-02-01", "draw"),
    ]
    train, test = split_using_poisson_policy(rows, date(2026, 2, 1))
    assert [row["provider_match_id"] for row in train] == [1]
    assert [row["provider_match_id"] for row in test] == [2, 3]
    assert max(row["match_date"] for row in train) < min(row["match_date"] for row in test)


def test_preprocessing_fits_training_rows_only_and_handles_unknown_categories():
    training = sample_training_rows()
    preprocessor = FittedPreprocessor.fit(training)
    before = preprocessor.to_dict()
    test_row = make_row(50, "2026-02-01", "home_win", home_team="Future Club")
    test_row["home_previous_goals_scored"] = 99999
    transformed = preprocessor.transform([test_row])[0]

    assert "Future Club" not in preprocessor.categorical_values["home_team"]
    assert preprocessor.to_dict() == before
    assert len(transformed) == len(preprocessor.feature_names)
    assert preprocessor.numeric_stats["home_previous_goals_scored"]["mean"] != 99999


def test_xgboost_probabilities_are_valid_and_sum_to_one():
    training = sample_training_rows()
    preprocessor, classifier = fit_xgboost(training)
    test = [make_row(20, "2026-02-01", "draw")]
    metrics, predictions = evaluate_classifier(test, preprocessor, classifier)

    for prediction in predictions:
        probabilities = [prediction[key] for key in (
            "home_win_probability", "draw_probability", "away_win_probability",
        )]
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)
    assert set(metrics["per_class"]) == {"home_win", "draw", "away_win"}


def test_training_and_evaluation_are_deterministic():
    training = sample_training_rows()
    test = [make_row(20, "2026-02-01", "draw"), make_row(21, "2026-02-02", "away_win")]
    preprocessor_a, classifier_a = fit_xgboost(training)
    preprocessor_b, classifier_b = fit_xgboost(training)
    metrics_a, predictions_a = evaluate_classifier(test, preprocessor_a, classifier_a)
    metrics_b, predictions_b = evaluate_classifier(test, preprocessor_b, classifier_b)

    assert metrics_a == metrics_b
    assert predictions_a == predictions_b
    assert preprocessor_a.to_dict() == preprocessor_b.to_dict()


def test_test_match_ids_must_exactly_align_with_both_baselines(tmp_path):
    rows = [make_row(100, "2026-10-31", "home_win"), make_row(101, "2026-11-01", "draw")]
    poisson = tmp_path / "poisson.csv"
    logistic = tmp_path / "logistic.csv"
    for path, ids in ((poisson, ("100", "101")), (logistic, ("100", "101"))):
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=["provider_match_id"])
            writer.writeheader()
            writer.writerows({"provider_match_id": match_id} for match_id in ids)
    verify_test_match_ids(rows, poisson, logistic)

    with poisson.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["provider_match_id"])
        writer.writeheader()
        writer.writerow({"provider_match_id": "100"})
    with pytest.raises(ValueError, match="Poisson"):
        verify_test_match_ids(rows, poisson, logistic)

    with poisson.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["provider_match_id"])
        writer.writeheader()
        writer.writerows({"provider_match_id": match_id} for match_id in ("100", "101"))
    with logistic.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["provider_match_id"])
        writer.writeheader()
        writer.writerow({"provider_match_id": "100"})
    with pytest.raises(ValueError, match="Logistic Regression"):
        verify_test_match_ids(rows, poisson, logistic)
