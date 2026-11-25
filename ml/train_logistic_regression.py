from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import sklearn

from ml.baselines.logistic_regression import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FittedPreprocessor,
    evaluate_classifier,
    fit_classifier,
    split_using_poisson_policy,
)
from ml.train_poisson_baseline import (
    DEFAULT_CUTOFF,
    DEFAULT_DATASET,
    _date_range,
    load_dataset,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1"
MODEL_DIRECTORY = REPOSITORY_ROOT / "ml" / "models" / "logistic_regression" / VERSION
EVALUATION_DIRECTORY = REPOSITORY_ROOT / "ml" / "evaluation" / "logistic_regression" / VERSION
POISSON_EVALUATION = REPOSITORY_ROOT / "ml" / "evaluation" / "poisson_baseline" / "v1" / "evaluation.json"
POISSON_PREDICTIONS = REPOSITORY_ROOT / "ml" / "evaluation" / "poisson_baseline" / "v1" / "test_predictions.csv"


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


def _verify_probabilities(predictions: list[dict[str, Any]]) -> None:
    columns = ("home_win_probability", "draw_probability", "away_win_probability")
    for row in predictions:
        values = [float(row[column]) for column in columns]
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
            raise ValueError(f"Invalid 1X2 probability for match {row['provider_match_id']}")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-10):
            raise ValueError(f"1X2 probabilities do not sum to 1 for match {row['provider_match_id']}")


def _assert_same_test_set(rows: list[dict[str, Any]]) -> None:
    with POISSON_PREDICTIONS.open(newline="", encoding="utf-8") as predictions_file:
        poisson_rows = list(csv.DictReader(predictions_file))
    expected_ids = [str(row["provider_match_id"]) for row in rows]
    actual_ids = [row["provider_match_id"] for row in poisson_rows]
    if expected_ids != actual_ids:
        raise ValueError("Logistic Regression test rows differ from the saved Poisson test set")


def _comparison(logistic_metrics: dict[str, Any]) -> dict[str, Any]:
    poisson = json.loads(POISSON_EVALUATION.read_text(encoding="utf-8"))
    poisson_metrics = poisson["metrics"]["one_x_two"]
    logistic = logistic_metrics
    return {
        "test_cutoff": poisson["cutoff_date"],
        "test_rows": poisson["test_rows"],
        "poisson_baseline_v1": {
            "log_loss": poisson_metrics["log_loss"],
            "brier_score": poisson_metrics["brier_score"],
            "accuracy": poisson_metrics["accuracy"],
        },
        "logistic_regression_v1": {
            "log_loss": logistic["log_loss"],
            "brier_score": logistic["brier_score"],
            "accuracy": logistic["accuracy"],
        },
        "delta_logistic_minus_poisson": {
            "log_loss": logistic["log_loss"] - poisson_metrics["log_loss"],
            "brier_score": logistic["brier_score"] - poisson_metrics["brier_score"],
            "accuracy": logistic["accuracy"] - poisson_metrics["accuracy"],
        },
    }


def _write_comparison(path: Path, comparison: dict[str, Any]) -> None:
    poisson = comparison["poisson_baseline_v1"]
    logistic = comparison["logistic_regression_v1"]
    report = "\n".join([
        "# Logistic Regression vs Poisson",
        "",
        f"Same chronological test set: {comparison['test_rows']} matches; cutoff {comparison['test_cutoff']}.",
        "",
        "| Model | Log Loss | Brier | Accuracy |",
        "|---|---:|---:|---:|",
        f"| Poisson v1 | {poisson['log_loss']:.4f} | {poisson['brier_score']:.4f} | {poisson['accuracy']:.2%} |",
        f"| Logistic Regression v1 | {logistic['log_loss']:.4f} | {logistic['brier_score']:.4f} | {logistic['accuracy']:.2%} |",
        "",
        "Lower Log Loss and Brier are better; higher accuracy is better.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run(dataset_path: Path = DEFAULT_DATASET, cutoff_date: date = DEFAULT_CUTOFF) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    rows = load_dataset(dataset_path)
    training_rows, test_rows = split_using_poisson_policy(rows, cutoff_date)
    if cutoff_date != DEFAULT_CUTOFF:
        raise ValueError("The comparison baseline requires the exact Poisson cutoff date")
    _assert_same_test_set(test_rows)

    preprocessor, classifier = fit_classifier(training_rows)
    metrics, predictions = evaluate_classifier(test_rows, preprocessor, classifier)
    _verify_probabilities(predictions)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    comparison = _comparison(metrics)

    model_metadata = {
        "model_version": VERSION,
        "model_type": "multinomial_logistic_regression_1x2",
        "scikit_learn_version": sklearn.__version__,
        "dataset_sha256": dataset_hash,
        "cutoff_date": cutoff_date.isoformat(),
        "training_rows": len(training_rows),
        "training_date_range": _date_range(training_rows),
        "classes": {"0": "home_win", "1": "draw", "2": "away_win"},
        "features": {
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
            "transformed_feature_count": len(preprocessor.feature_names),
        },
        "logistic_regression_parameters": {
            "C": classifier.C,
            "solver": classifier.solver,
            "max_iter": classifier.max_iter,
            "random_state": classifier.random_state,
            "class_weight": classifier.class_weight,
        },
    }
    evaluation = {
        "model_version": VERSION,
        "dataset_sha256": dataset_hash,
        "cutoff_date": cutoff_date.isoformat(),
        "split_policy": "same Poisson policy: train match_date < 2026-10-31; test match_date >= 2026-10-31; no shuffle",
        "training_rows": len(training_rows),
        "test_rows": len(test_rows),
        "training_date_range": _date_range(training_rows),
        "test_date_range": _date_range(test_rows),
        "metrics": metrics,
        "poisson_comparison": comparison,
        "probability_validation": {
            "all_1x2_probabilities_in_0_to_1": True,
            "all_1x2_distributions_sum_to_1": True,
            "test_rows_checked": len(predictions),
            "test_match_ids_match_poisson": True,
        },
        "limitations": [
            "Small, provider-limited sample; team/category coverage is sparse.",
            "Unknown test-time teams are represented by all-zero one-hot columns.",
            "Logistic Regression models 1X2 only and does not predict goals or goal markets.",
            "This is a benchmark, not a production-ready model.",
        ],
    }

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    model_temp = MODEL_DIRECTORY / "model.joblib.tmp"
    try:
        joblib.dump(classifier, model_temp)
        model_temp.replace(MODEL_DIRECTORY / "model.joblib")
    finally:
        model_temp.unlink(missing_ok=True)
    _write_json(MODEL_DIRECTORY / "model_metadata.json", model_metadata)
    _write_json(MODEL_DIRECTORY / "preprocessing.json", preprocessor.to_dict())
    _write_json(EVALUATION_DIRECTORY / "evaluation.json", evaluation)
    _write_csv(EVALUATION_DIRECTORY / "test_predictions.csv", predictions)
    _write_comparison(EVALUATION_DIRECTORY / "comparison_with_poisson.md", comparison)
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the chronological Logistic Regression baseline")
    parser.add_argument("--dataset", dest="dataset_path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cutoff-date", type=date.fromisoformat, default=DEFAULT_CUTOFF)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(**vars(parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
