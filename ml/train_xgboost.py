from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import xgboost

from ml.baselines.logistic_regression import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    evaluate_classifier,
    split_using_poisson_policy,
)
from ml.baselines.xgboost_classifier import XGBOOST_PARAMETERS, fit_xgboost
from ml.train_logistic_regression import (
    DEFAULT_CUTOFF,
    DEFAULT_DATASET,
    POISSON_PREDICTIONS,
)
from ml.train_poisson_baseline import (
    EVALUATION_DIRECTORY as POISSON_EVALUATION_DIRECTORY,
    _date_range,
    load_dataset,
)
from ml.train_logistic_regression import EVALUATION_DIRECTORY as LOGISTIC_EVALUATION_DIRECTORY

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1"
MODEL_DIRECTORY = REPOSITORY_ROOT / "ml" / "models" / "xgboost" / VERSION
EVALUATION_DIRECTORY = REPOSITORY_ROOT / "ml" / "evaluation" / "xgboost" / VERSION
POISSON_EVALUATION = POISSON_EVALUATION_DIRECTORY / "evaluation.json"
LOGISTIC_EVALUATION = LOGISTIC_EVALUATION_DIRECTORY / "evaluation.json"
LOGISTIC_PREDICTIONS = LOGISTIC_EVALUATION_DIRECTORY / "test_predictions.csv"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_test_match_ids(
    test_rows: list[dict[str, Any]],
    poisson_predictions_path: Path = POISSON_PREDICTIONS,
    logistic_predictions_path: Path = LOGISTIC_PREDICTIONS,
) -> None:
    expected_ids = [str(row["provider_match_id"]) for row in test_rows]
    for artifact_path, model_name in (
        (poisson_predictions_path, "Poisson"),
        (logistic_predictions_path, "Logistic Regression"),
    ):
        with artifact_path.open(newline="", encoding="utf-8") as predictions_file:
            saved_ids = [row["provider_match_id"] for row in csv.DictReader(predictions_file)]
        if saved_ids != expected_ids:
            raise ValueError(f"XGBoost test rows do not match the saved {model_name} test set")


def _validate_probabilities(predictions: list[dict[str, Any]]) -> None:
    columns = ("home_win_probability", "draw_probability", "away_win_probability")
    for row in predictions:
        values = [float(row[column]) for column in columns]
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError(f"Invalid XGBoost probability for match {row['provider_match_id']}")
        # XGBoost emits float32 probabilities, so allow normal float32 rounding.
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError(f"XGBoost 1X2 probabilities do not sum to 1 for match {row['provider_match_id']}")


def _comparison(xgboost_metrics: dict[str, Any]) -> dict[str, Any]:
    poisson = json.loads(POISSON_EVALUATION.read_text(encoding="utf-8"))["metrics"]["one_x_two"]
    logistic = json.loads(LOGISTIC_EVALUATION.read_text(encoding="utf-8"))["metrics"]
    xgb = xgboost_metrics
    return {
        "test_cutoff": DEFAULT_CUTOFF.isoformat(),
        "test_rows": 143,
        "models": {
            "Poisson v1": {"log_loss": poisson["log_loss"], "brier_score": poisson["brier_score"], "accuracy": poisson["accuracy"]},
            "Logistic Regression v1": {"log_loss": logistic["log_loss"], "brier_score": logistic["brier_score"], "accuracy": logistic["accuracy"]},
            "XGBoost v1": {"log_loss": xgb["log_loss"], "brier_score": xgb["brier_score"], "accuracy": xgb["accuracy"]},
        },
    }


def _write_comparison(path: Path, comparison: dict[str, Any]) -> None:
    models = comparison["models"]
    report = "\n".join([
        "# Baseline model comparison",
        "",
        f"Same chronological test set: {comparison['test_rows']} matches; cutoff {comparison['test_cutoff']}.",
        "",
        "| Model | Log Loss | Brier | Accuracy |",
        "|---|---:|---:|---:|",
        *[
            f"| {name} | {values['log_loss']:.4f} | {values['brier_score']:.4f} | {values['accuracy']:.2%} |"
            for name, values in models.items()
        ],
        "",
        "Lower Log Loss and Brier are better; higher accuracy is better. Consider probability quality as well as accuracy.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run(dataset_path: Path = DEFAULT_DATASET, cutoff_date: date = DEFAULT_CUTOFF) -> dict[str, Any]:
    if cutoff_date != DEFAULT_CUTOFF:
        raise ValueError("XGBoost comparison must use the exact existing chronological cutoff")
    dataset_path = dataset_path.resolve()
    rows = load_dataset(dataset_path)
    training_rows, test_rows = split_using_poisson_policy(rows, cutoff_date)
    if len(training_rows) != 515 or len(test_rows) != 143:
        raise ValueError("Dataset no longer matches the verified 515/143 Poisson and Logistic split")
    verify_test_match_ids(test_rows)

    preprocessor, classifier = fit_xgboost(training_rows)
    metrics, predictions = evaluate_classifier(test_rows, preprocessor, classifier)
    _validate_probabilities(predictions)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    comparison = _comparison(metrics)
    model_metadata = {
        "model_version": VERSION,
        "model_type": "xgboost_multiclass_1x2",
        "xgboost_version": xgboost.__version__,
        "dataset_sha256": dataset_hash,
        "cutoff_date": cutoff_date.isoformat(),
        "training_rows": len(training_rows),
        "training_date_range": _date_range(training_rows),
        "classes": {"0": "home_win", "1": "draw", "2": "away_win"},
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "transformed_feature_count": len(preprocessor.feature_names),
        "parameters": XGBOOST_PARAMETERS,
        "preprocessing_fit_rows": len(training_rows),
    }
    evaluation = {
        "model_version": VERSION,
        "dataset_sha256": dataset_hash,
        "cutoff_date": cutoff_date.isoformat(),
        "split_policy": "same existing split: train match_date < 2026-10-31; test match_date >= 2026-10-31; no shuffle",
        "training_rows": len(training_rows),
        "test_rows": len(test_rows),
        "training_date_range": _date_range(training_rows),
        "test_date_range": _date_range(test_rows),
        "metrics": metrics,
        "comparison": comparison,
        "probability_validation": {
            "all_probabilities_in_0_to_1": True,
            "all_1x2_distributions_sum_to_1": True,
            "test_rows_checked": len(predictions),
            "test_ids_match_poisson": True,
            "test_ids_match_logistic_regression": True,
        },
        "limitations": [
            "Small provider-limited dataset relative to the one-hot feature space.",
            "Conservative fixed configuration was not hyperparameter tuned.",
            "This is a benchmark, not a production-ready model.",
        ],
    }

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_temp = MODEL_DIRECTORY / "model.tmp.json"
    try:
        classifier.save_model(str(model_temp))
        model_temp.replace(MODEL_DIRECTORY / "model.json")
    finally:
        model_temp.unlink(missing_ok=True)
    _write_json(MODEL_DIRECTORY / "model_metadata.json", model_metadata)
    _write_json(MODEL_DIRECTORY / "preprocessing.json", preprocessor.to_dict())
    _write_json(EVALUATION_DIRECTORY / "evaluation.json", evaluation)
    _write_csv(EVALUATION_DIRECTORY / "test_predictions.csv", predictions)
    _write_comparison(EVALUATION_DIRECTORY / "comparison.md", comparison)
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the chronological XGBoost 1X2 baseline")
    parser.add_argument("--dataset", dest="dataset_path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cutoff-date", type=date.fromisoformat, default=DEFAULT_CUTOFF)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(**vars(parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
