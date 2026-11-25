import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml.features.feature_engineering_v2 import FEATURE_COLUMNS, TARGET_COLUMNS
from ml.train_poisson_comparison import (
    _fit_c_model,
    _score_grid,
    evaluate_prediction_rows,
    preflight,
)
from ml.training.model_outputs import assert_unprotected_output_path

ROOT = Path(__file__).resolve().parents[2]


def test_preflight_confirms_exact_schemas_splits_award_and_folds():
    report = preflight()
    assert report["status"] == "PASS"
    assert report["comparator_b_feature_count"] == 6
    assert report["poisson_v2_feature_count"] == 13
    assert report["training_count"] == 2556
    assert report["test_count"] == 143
    assert report["excluded_fixture_ids"] == [542704]
    assert [fold["name"] for fold in report["folds"]] == ["fold_1", "fold_2", "fold_3", "fold_4"]
    assert [fold["validation_rows"] for fold in report["folds"]] == [682, 421, 313, 186]


def test_v2_targets_and_metadata_are_not_model_features():
    forbidden = {"provider_match_id", "match_date", "competition", "home_team", "away_team", *TARGET_COLUMNS}
    assert not set(FEATURE_COLUMNS) & forbidden
    assert len(FEATURE_COLUMNS) == 13


def test_score_grid_emits_valid_probabilities_and_prediction():
    prediction = _score_grid(1.55, 1.1)
    probabilities = [prediction[key] for key in ("home_win_probability", "draw_probability", "away_win_probability")]
    assert all(0 <= probability <= 1 for probability in probabilities)
    assert abs(sum(probabilities) - 1) < 1e-10
    assert isinstance(prediction["predicted_home_score"], int)
    assert prediction["over_1_5_probability"] >= prediction["over_2_5_probability"]
    assert prediction["over_2_5_probability"] >= prediction["over_3_5_probability"]


def test_evaluation_reports_class_calibration_and_separate_goal_errors():
    prediction = {
        "provider_match_id": 1,
        "home_win_probability": 0.55, "draw_probability": 0.25, "away_win_probability": 0.20,
        "expected_home_goals": 1.6, "expected_away_goals": 1.0,
        "actual_home_win": 1, "actual_draw": 0, "actual_away_win": 0,
        "actual_home_goals": 2, "actual_away_goals": 0, "actual_total_goals": 2,
    }
    metrics = evaluate_prediction_rows([prediction])
    assert metrics["sample_count"] == 1
    assert set(metrics["one_x_two"]["classwise_calibration"]) == {"home_win", "draw", "away_win"}
    assert metrics["goals"]["home_goals"]["mae"] == pytest.approx(0.4)
    assert metrics["goals"]["away_goals"]["mae"] == pytest.approx(1.0)
    assert metrics["goals"]["total_goals"]["mae"] == pytest.approx(0.6)


def test_v2_model_and_probabilities_are_deterministic_for_fixed_training_rows():
    training = []
    for index in range(40):
        row = {feature: float((index + offset) % 7) for offset, feature in enumerate(FEATURE_COLUMNS)}
        row.update(home_goals=index % 4, away_goals=(index + 1) % 3)
        training.append(row)
    preprocessing_one, home_one, away_one = _fit_c_model(training)
    preprocessing_two, home_two, away_two = _fit_c_model(training)
    assert np.allclose(home_one.coef_, home_two.coef_)
    assert np.allclose(away_one.coef_, away_two.coef_)
    assert preprocessing_one.fitted_state() == preprocessing_two.fitted_state()
    matrix = preprocessing_one.transform(training[:1])
    prediction_one = _score_grid(home_one.predict(matrix)[0], away_one.predict(matrix)[0])
    prediction_two = _score_grid(home_two.predict(matrix)[0], away_two.predict(matrix)[0])
    assert prediction_one == prediction_two


def test_training_outputs_cannot_target_v1_artifacts():
    with pytest.raises(ValueError, match="protected v1"):
        assert_unprotected_output_path(ROOT / "ml/models/poisson_baseline/v1/model.json")


def test_frozen_v1_and_v2_data_hashes_are_unchanged():
    expected = {
        "ml/models/poisson_baseline/v1/model.json": "77b138a6014601b19ee32021da3ce80039cafe82270d8d890b524636d3facd1a",
        "ml/evaluation/poisson_baseline/v1/evaluation.json": "9965f5209a9630fc2bac23946e9cf4e85b017ad6cba65cc9bdd6be053d4ad41b",
        "ml/evaluation/poisson_baseline/v1/test_predictions.csv": "cf61138fbf45bba36dbcc3e9df8f73fd2b634584867b2fe2554766ce1afcea7e",
        "ml/data/processed/v2/training_dataset.csv": "7656a4694f315f02913e55260d7599f7034ab27b5a671ca10ffd902375a461d5",
        "ml/data/processed/v2/manifest.json": "67daae2ea0361ccdc8ede1f1630b9cf0e518068dee0103f9fc040733b59317e6",
    }
    for relative, expected_hash in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected_hash

