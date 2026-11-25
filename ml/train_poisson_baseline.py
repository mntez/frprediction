from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

from ml.baselines.poisson import (
    TEAM_FEATURES,
    PoissonBaseline,
    evaluate_predictions,
    split_chronologically,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1"
DEFAULT_CUTOFF = date(2026, 10, 31)
DEFAULT_DATASET = REPOSITORY_ROOT / "ml" / "data" / "processed" / "training_dataset.csv"
MODEL_DIRECTORY = REPOSITORY_ROOT / "ml" / "models" / "poisson_baseline" / VERSION
EVALUATION_DIRECTORY = REPOSITORY_ROOT / "ml" / "evaluation" / "poisson_baseline" / VERSION


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        if not reader.fieldnames:
            raise ValueError("Training dataset has no CSV header")
        required = {
            "provider_match_id", "match_date", "competition", "home_team", "away_team",
            "home_goals", "away_goals", "home_win", "draw", "away_win", "total_goals",
            "over_1_5", "over_2_5", "over_3_5", "btts", *TEAM_FEATURES,
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError("Training dataset is missing columns: " + ", ".join(missing))
        rows = list(reader)
    if not rows:
        raise ValueError("Training dataset contains no rows")
    return rows


def _date_range(rows: list[dict[str, Any]]) -> dict[str, str]:
    dates = [date.fromisoformat(str(row["match_date"])) for row in rows]
    return {"start": min(dates).isoformat(), "end": max(dates).isoformat()}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
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


def validate_probabilities(rows: list[dict[str, Any]]) -> None:
    probability_columns = (
        "home_win_probability", "draw_probability", "away_win_probability",
        "over_1_5_probability", "over_2_5_probability", "over_3_5_probability",
        "btts_probability",
    )
    for row in rows:
        probabilities = [float(row[column]) for column in probability_columns]
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
            raise ValueError(f"Invalid probability for test match {row['provider_match_id']}")
        if not math.isclose(sum(float(row[key]) for key in probability_columns[:3]), 1.0, abs_tol=1e-10):
            raise ValueError(f"1X2 probabilities do not sum to 1 for match {row['provider_match_id']}")


def run(dataset_path: Path = DEFAULT_DATASET, cutoff_date: date = DEFAULT_CUTOFF) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    source_rows = load_dataset(dataset_path)
    training_rows, test_rows = split_chronologically(source_rows, cutoff_date)
    model = PoissonBaseline.fit(training_rows)
    metrics, predictions = evaluate_predictions(test_rows, model)
    validate_probabilities(predictions)

    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    model_artifact = {
        "model_version": VERSION,
        "model_type": "independent_team_strength_poisson",
        "dataset_sha256": dataset_hash,
        "cutoff_date": cutoff_date.isoformat(),
        "training_rows": len(training_rows),
        "training_date_range": _date_range(training_rows),
        "training_goal_means": {
            "home": model.home_goal_mean,
            "away": model.away_goal_mean,
        },
        "team_features_used": list(TEAM_FEATURES),
        "model_parameters": model.to_dict(),
        "methodology": {
            "prior_matches": model.prior_matches,
            "goal_rate_clamp": [model.minimum_goal_rate, model.maximum_goal_rate],
            "score_probability_grid_max_goals": model.max_goals,
            "team_rates": "Venue-specific attack and defence goal averages shrunk toward training-period league means.",
        },
    }
    evaluation = {
        "model_version": VERSION,
        "dataset_sha256": dataset_hash,
        "cutoff_date": cutoff_date.isoformat(),
        "split_policy": "train match_date < cutoff; test match_date >= cutoff; no shuffled rows",
        "training_rows": len(training_rows),
        "test_rows": len(test_rows),
        "training_date_range": _date_range(training_rows),
        "test_date_range": _date_range(test_rows),
        "metrics": metrics,
        "probability_validation": {
            "all_probability_values_in_0_to_1": True,
            "all_1x2_distributions_sum_to_1": True,
            "test_rows_checked": len(predictions),
        },
        "limitations": [
            "Small, provider-limited dataset with sparse earlier history.",
            "Independent Poisson goals omit correlation and changing team strength over time.",
            "The test uses pre-match rolling features, including earlier test-period results only after they occurred.",
            "This baseline is for comparison and is not production-ready.",
        ],
    }

    _write_json(MODEL_DIRECTORY / "model.json", model_artifact)
    _write_json(EVALUATION_DIRECTORY / "evaluation.json", evaluation)
    _write_predictions(EVALUATION_DIRECTORY / "test_predictions.csv", predictions)
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and chronologically evaluate the Poisson baseline")
    parser.add_argument("--dataset", dest="dataset_path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cutoff-date", type=date.fromisoformat, default=DEFAULT_CUTOFF)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(**vars(parse_args()))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
