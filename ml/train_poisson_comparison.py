"""Train and evaluate isolated expanded-history Poisson B and redesigned v2 C."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import warnings
from datetime import date
from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import PoissonRegressor

from ml.baselines.poisson import TEAM_FEATURES, PoissonBaseline, poisson_probabilities, target_labels
from ml.features.comparator_b import CUTOFF_DATE as DEFAULT_CUTOFF, EXCLUDED_FIXTURE_IDS
from ml.features.feature_engineering_v2 import FEATURE_COLUMNS as V2_FEATURES, TARGET_COLUMNS
from ml.training.fold_preprocessing import (
    ROLLING_ORIGIN_FOLDS,
    FoldPreprocessor,
    prepare_rolling_fold,
    split_rolling_fold,
)
from ml.training.model_outputs import assert_unprotected_output_path, get_output_directory

ROOT = Path(__file__).resolve().parents[1]
B_DATASET = ROOT / "ml/data/processed/comparator_b/v1/training_dataset.csv"
B_MANIFEST = B_DATASET.with_name("manifest.json")
C_DATASET = ROOT / "ml/data/processed/v2/training_dataset.csv"
C_MANIFEST = C_DATASET.with_name("manifest.json")
A_EVALUATION = ROOT / "ml/evaluation/poisson_baseline/v1/evaluation.json"
A_PREDICTIONS = ROOT / "ml/evaluation/poisson_baseline/v1/test_predictions.csv"
CLASS_KEYS = ("home_win", "draw", "away_win")
CLASS_PROBABILITY_KEYS = ("home_win_probability", "draw_probability", "away_win_probability")
GOAL_RATE_MIN = 0.05
GOAL_RATE_MAX = 8.0
GRID_MAX_GOALS = 32
CALIBRATION_BINS = 10
POISSON_V2_SETTINGS = {
    "estimator": "sklearn.linear_model.PoissonRegressor",
    "alpha": 1.0,
    "fit_intercept": True,
    "max_iter": 1000,
    "tol": 1e-7,
    "link": "log",
    "goal_rate_clamp": [GOAL_RATE_MIN, GOAL_RATE_MAX],
    "score_probability_grid_max_goals": GRID_MAX_GOALS,
    "feature_selection": "none",
}


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"Dataset has no header: {path}")
        return list(reader)


def _number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Dataset contains a non-finite model input")
    return result


def _typed_rows(rows: list[dict[str, str]], features: tuple[str, ...]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = dict(row)
        item["provider_match_id"] = int(row["provider_match_id"])
        for name in features:
            item[name] = _number(row.get(name))
        for name in TARGET_COLUMNS:
            item[name] = int(row[name])
        converted.append(item)
    return converted


def _ordered_ids(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["provider_match_id"]) for row in rows]


def preflight() -> dict[str, Any]:
    """Fail closed on schema, IDs, eligibility, hashes or fold drift before fitting."""
    b_manifest = json.loads(B_MANIFEST.read_text(encoding="utf-8"))
    c_manifest = json.loads(C_MANIFEST.read_text(encoding="utf-8"))
    b_raw, c_raw = _read_csv(B_DATASET), _read_csv(C_DATASET)
    b_hash, c_hash = _file_hash(B_DATASET), _file_hash(C_DATASET)
    if b_hash != b_manifest["dataset_sha256"] or c_hash != c_manifest["dataset_sha256"]:
        raise ValueError("A B/C dataset hash does not match its manifest")
    if tuple(b_manifest["feature_columns"]) != TEAM_FEATURES:
        raise ValueError("Comparator B is not using the exact original six features")
    if tuple(c_manifest["feature_columns"]) != tuple(V2_FEATURES) or len(V2_FEATURES) != 13:
        raise ValueError("Poisson v2 feature schema differs from the approved 13 features")
    if set(TEAM_FEATURES) & set(TARGET_COLUMNS) or set(V2_FEATURES) & set(TARGET_COLUMNS):
        raise ValueError("A target is present in a model input feature list")
    b_ids, c_ids = _ordered_ids(b_raw), _ordered_ids(c_raw)
    if b_ids != c_ids or b_ids != [int(value) for value in c_manifest["ordered_fixture_ids"]]:
        raise ValueError("B and C ordered fixture IDs are not identical")
    b_split, c_split = b_manifest["split"], c_manifest["split"]
    if b_split["training_fixture_ids"] != c_split["training_fixture_ids"] or b_split["test_fixture_ids"] != c_split["test_fixture_ids"]:
        raise ValueError("B and C train/test fixture manifests differ")
    if len(b_split["training_fixture_ids"]) != 2556 or len(b_split["test_fixture_ids"]) != 143:
        raise ValueError("Approved B/C split row counts changed")
    if EXCLUDED_FIXTURE_IDS & set(b_ids):
        raise ValueError("Awarded fixture 542704 appears in the B/C datasets")
    if b_manifest.get("excluded_awarded_fixture_ids") != sorted(EXCLUDED_FIXTURE_IDS):
        raise ValueError("B manifest does not document the awarded-fixture exclusion")
    v1_predictions = _read_csv(A_PREDICTIONS)
    v1_test_ids = [int(row["provider_match_id"]) for row in v1_predictions]
    if [int(value) for value in c_split["test_fixture_ids"]] != v1_test_ids:
        raise ValueError("B/C test fixture IDs differ from the frozen v1 benchmark")
    if len(set(b_ids)) != len(b_ids) or len(set(c_ids)) != len(c_ids):
        raise ValueError("Duplicate fixture IDs found")
    output_paths = {
        "B_model": get_output_directory("comparator_b", "model"),
        "B_evaluation": get_output_directory("comparator_b", "evaluation"),
        "C_model": get_output_directory("poisson_v2", "model"),
        "C_evaluation": get_output_directory("poisson_v2", "evaluation"),
    }
    for raw_rows, inputs, label in ((b_raw, TEAM_FEATURES, "B"), (c_raw, V2_FEATURES, "C")):
        if any(set(inputs) - set(raw_rows[0]) for _ in (0,)):
            raise ValueError(f"{label} dataset is missing model inputs")
        forbidden = {"provider_match_id", "match_date", "competition", "home_team", "away_team", *TARGET_COLUMNS}
        if set(inputs) & forbidden:
            raise ValueError(f"{label} uses identifier/date/target/post-match fields as inputs")
    b_rows = _typed_rows(b_raw, TEAM_FEATURES)
    c_rows = _typed_rows(c_raw, tuple(V2_FEATURES))
    fold_summary = []
    for fold in ROLLING_ORIGIN_FOLDS:
        b_train, b_validation = split_rolling_fold(b_rows, fold)
        c_train, c_validation = split_rolling_fold(c_rows, fold)
        if _ordered_ids(b_train) != _ordered_ids(c_train) or _ordered_ids(b_validation) != _ordered_ids(c_validation):
            raise ValueError(f"B and C fixture IDs differ within {fold.name}")
        if max(row["match_date"] for row in b_train) >= min(row["match_date"] for row in b_validation):
            raise ValueError(f"Fold {fold.name} overlaps train and validation dates")
        fold_summary.append({
            "name": fold.name, "train_rows": len(b_train), "validation_rows": len(b_validation),
            "train_end": fold.train_end, "validation_start": fold.validation_start,
            "validation_end": fold.validation_end,
        })
    return {
        "status": "PASS",
        "comparator_b_feature_count": len(TEAM_FEATURES),
        "poisson_v2_feature_count": len(V2_FEATURES),
        "row_count": len(b_ids),
        "training_count": len(b_split["training_fixture_ids"]),
        "test_count": len(b_split["test_fixture_ids"]),
        "excluded_fixture_ids": sorted(EXCLUDED_FIXTURE_IDS),
        "ordered_training_ids_sha256": b_split["training_fixture_ids_sha256"],
        "ordered_test_ids_sha256": b_split["test_fixture_ids_sha256"],
        "b_dataset_sha256": b_hash,
        "c_dataset_sha256": c_hash,
        "folds": fold_summary,
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "v1_model_config": json.loads((ROOT / "ml/models/poisson_baseline/v1/model.json").read_text(encoding="utf-8")),
        "v1_metrics": json.loads(A_EVALUATION.read_text(encoding="utf-8"))["metrics"],
    }


def _validate_targets(row: dict[str, Any]) -> None:
    expected = target_labels(int(row["home_goals"]), int(row["away_goals"]))
    for key, value in expected.items():
        if int(row[key]) != value:
            raise ValueError(f"Target {key} disagrees with stored score for fixture {row['provider_match_id']}")


def _score_grid(home_rate: float, away_rate: float) -> dict[str, float | int]:
    home_rate = min(max(float(home_rate), GOAL_RATE_MIN), GOAL_RATE_MAX)
    away_rate = min(max(float(away_rate), GOAL_RATE_MIN), GOAL_RATE_MAX)
    home_pmf = poisson_probabilities(home_rate, max_goals=GRID_MAX_GOALS)
    away_pmf = poisson_probabilities(away_rate, max_goals=GRID_MAX_GOALS)
    h, d, a = 0.0, 0.0, 0.0
    overs = {1.5: 0.0, 2.5: 0.0, 3.5: 0.0}
    most_likely = (-1.0, 0, 0)
    for hg, hp in enumerate(home_pmf):
        for ag, ap in enumerate(away_pmf):
            p = hp * ap
            if hg > ag:
                h += p
            elif hg == ag:
                d += p
            else:
                a += p
            total = hg + ag
            for line in overs:
                if total > line:
                    overs[line] += p
            if p > most_likely[0]:
                most_likely = (p, hg, ag)
    return {
        "expected_home_goals": home_rate,
        "expected_away_goals": away_rate,
        "home_win_probability": h,
        "draw_probability": d,
        "away_win_probability": a,
        "over_1_5_probability": overs[1.5],
        "over_2_5_probability": overs[2.5],
        "over_3_5_probability": overs[3.5],
        "btts_probability": (1 - home_pmf[0]) * (1 - away_pmf[0]),
        "predicted_home_score": most_likely[1],
        "predicted_away_score": most_likely[2],
    }


def _prediction_row(source: dict[str, Any], model_prediction: dict[str, Any]) -> dict[str, Any]:
    _validate_targets(source)
    return {
        "provider_match_id": int(source["provider_match_id"]),
        "match_date": source["match_date"],
        "competition": source["competition"],
        "home_team": source["home_team"],
        "away_team": source["away_team"],
        **model_prediction,
        **{f"actual_{name}": int(source[name]) for name in TARGET_COLUMNS},
    }


def _calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    for outcome, probability_key in zip(CLASS_KEYS, CLASS_PROBABILITY_KEYS):
        samples = [(float(row[probability_key]), int(row[f"actual_{outcome}"])) for row in rows]
        bins = []
        ece = 0.0
        for index in range(CALIBRATION_BINS):
            low, high = index / CALIBRATION_BINS, (index + 1) / CALIBRATION_BINS
            bucket = [pair for pair in samples if low <= pair[0] < high or (index == CALIBRATION_BINS - 1 and pair[0] == 1.0)]
            if not bucket:
                continue
            mean_probability = sum(value for value, _ in bucket) / len(bucket)
            observed_rate = sum(actual for _, actual in bucket) / len(bucket)
            ece += len(bucket) / len(samples) * abs(mean_probability - observed_rate)
            bins.append({
                "lower_bound": low, "upper_bound": high, "count": len(bucket),
                "mean_probability": mean_probability, "observed_rate": observed_rate,
            })
        per_class[outcome] = {"expected_calibration_error": ece, "bins": bins}
    return per_class


def evaluate_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot evaluate an empty prediction set")
    log_loss = brier = correct = 0.0
    home_errors, away_errors, total_errors = [], [], []
    for row in rows:
        probabilities = [float(row[key]) for key in CLASS_PROBABILITY_KEYS]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
            raise ValueError(f"Invalid probability for fixture {row['provider_match_id']}")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-10):
            raise ValueError(f"1X2 probabilities do not sum to 1 for fixture {row['provider_match_id']}")
        actual = [int(row[f"actual_{key}"]) for key in CLASS_KEYS]
        if sum(actual) != 1:
            raise ValueError(f"Invalid 1X2 target for fixture {row['provider_match_id']}")
        selected = next(i for i, value in enumerate(actual) if value)
        log_loss -= math.log(max(probabilities[selected], 1e-15))
        brier += sum((p - y) ** 2 for p, y in zip(probabilities, actual))
        correct += int(max(range(3), key=lambda i: probabilities[i]) == selected)
        home_errors.append(float(row["expected_home_goals"]) - int(row["actual_home_goals"]))
        away_errors.append(float(row["expected_away_goals"]) - int(row["actual_away_goals"]))
        total_errors.append(
            float(row["expected_home_goals"]) + float(row["expected_away_goals"]) - int(row["actual_total_goals"])
        )

    def regression_metrics(errors: list[float]) -> dict[str, float]:
        return {"mae": sum(abs(e) for e in errors) / len(errors), "rmse": math.sqrt(sum(e * e for e in errors) / len(errors))}

    return {
        "sample_count": len(rows),
        "one_x_two": {
            "log_loss": log_loss / len(rows),
            "brier_score": brier / len(rows),
            "accuracy": correct / len(rows),
            "classwise_calibration": _calibration(rows),
        },
        "goals": {
            "home_goals": regression_metrics(home_errors),
            "away_goals": regression_metrics(away_errors),
            "total_goals": regression_metrics(total_errors),
        },
    }


def _fit_c_model(
    training_rows: list[dict[str, Any]],
    preprocessor: FoldPreprocessor | None = None,
    training_matrix=None,
) -> tuple[FoldPreprocessor, PoissonRegressor, PoissonRegressor]:
    preprocessor = preprocessor or FoldPreprocessor(tuple(V2_FEATURES)).fit(training_rows)
    matrix = training_matrix if training_matrix is not None else preprocessor.transform(training_rows)
    home = PoissonRegressor(alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-7)
    away = PoissonRegressor(alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        home.fit(matrix, [int(row["home_goals"]) for row in training_rows])
        away.fit(matrix, [int(row["away_goals"]) for row in training_rows])
    preprocessor.fit_warnings = [str(item.message) for item in caught]  # type: ignore[attr-defined]
    return preprocessor, home, away


def _c_predict(rows: list[dict[str, Any]], preprocessor: FoldPreprocessor, home: PoissonRegressor, away: PoissonRegressor, matrix=None) -> list[dict[str, Any]]:
    matrix = matrix if matrix is not None else preprocessor.transform(rows)
    home_rates, away_rates = home.predict(matrix), away.predict(matrix)
    result = []
    for source, home_rate, away_rate in zip(rows, home_rates, away_rates):
        result.append(_prediction_row(source, _score_grid(home_rate, away_rate)))
    return result


def _b_predict(rows: list[dict[str, Any]], model: PoissonBaseline) -> list[dict[str, Any]]:
    return [_prediction_row(row, model.predict(row)) for row in rows]


def _date_range(rows: list[dict[str, Any]]) -> dict[str, str]:
    dates = [str(row["match_date"]) for row in rows]
    return {"start": min(dates), "end": max(dates)}


def _fold_manifest(train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> dict[str, Any]:
    train_ids, val_ids = _ordered_ids(train), _ordered_ids(validation)
    return {
        "training_count": len(train_ids), "training_fixture_ids": train_ids,
        "training_fixture_ids_sha256": _canonical_hash(train_ids),
        "validation_count": len(val_ids), "validation_fixture_ids": val_ids,
        "validation_fixture_ids_sha256": _canonical_hash(val_ids),
    }


def _safe_json(path: Path, payload: Any) -> None:
    path = assert_unprotected_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _safe_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty predictions: {path}")
    path = assert_unprotected_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _safe_joblib(path: Path, value: Any) -> None:
    path = assert_unprotected_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        joblib.dump(value, temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _a_saved_metrics() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stored = json.loads(A_EVALUATION.read_text(encoding="utf-8"))["metrics"]
    rows = _read_csv(A_PREDICTIONS)
    typed = []
    for raw in rows:
        row: dict[str, Any] = dict(raw)
        for key in (*CLASS_PROBABILITY_KEYS, "expected_home_goals", "expected_away_goals"):
            row[key] = float(row[key])
        for key in ("actual_home_win", "actual_draw", "actual_away_win", "actual_home_goals", "actual_away_goals", "actual_total_goals"):
            row[key] = int(row[key])
        typed.append(row)
    derived = evaluate_prediction_rows(typed)
    derived["one_x_two"]["log_loss"] = stored["one_x_two"]["log_loss"]
    derived["one_x_two"]["brier_score"] = stored["one_x_two"]["brier_score"]
    derived["one_x_two"]["accuracy"] = stored["one_x_two"]["accuracy"]
    return derived, rows


def _comparison_markdown(a: dict[str, Any], b: dict[str, Any], c: dict[str, Any], fold_metrics: dict[str, Any]) -> str:
    lines = [
        "# Poisson A/B/C Controlled Comparison", "",
        "A is the frozen historical v1 benchmark. B and C use the same expanded-history training IDs and locked test IDs.",
        "A's original training fixture IDs were not preserved, so A vs B is not a pure sample-size experiment.",
        "B vs C uses the same fixture population; differences reflect feature architecture and the necessary model formulation/preprocessing described in `training_config.json`.",
        "These results are research baselines, not production predictions.", "",
        "## Final locked test set", "",
        "| Model | N | Log Loss | Brier | Accuracy | Home MAE | Away MAE | Total MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (("A — Poisson v1 frozen", a), ("B — Expanded history", b), ("C — Redesigned v2", c)):
        lines.append(
            f"| {name} | {metrics['sample_count']} | {metrics['one_x_two']['log_loss']:.4f} | "
            f"{metrics['one_x_two']['brier_score']:.4f} | {metrics['one_x_two']['accuracy']:.2%} | "
            f"{metrics['goals']['home_goals']['mae']:.4f} | {metrics['goals']['away_goals']['mae']:.4f} | "
            f"{metrics['goals']['total_goals']['mae']:.4f} |"
        )
    lines += ["", "## Rolling-origin validation", "", "Metrics are in fold-specific `metrics.json` files; validation sample counts are recorded for each fold.", ""]
    lines += ["| Fold | B N | B Log Loss | B Brier | B Accuracy | C N | C Log Loss | C Brier | C Accuracy |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for fold_name, contents in fold_metrics.items():
        bm, cm = contents["B"], contents["C"]
        lines.append(
            f"| {fold_name} | {bm['sample_count']} | {bm['one_x_two']['log_loss']:.4f} | {bm['one_x_two']['brier_score']:.4f} | {bm['one_x_two']['accuracy']:.2%} | "
            f"{cm['sample_count']} | {cm['one_x_two']['log_loss']:.4f} | {cm['one_x_two']['brier_score']:.4f} | {cm['one_x_two']['accuracy']:.2%} |"
        )
    return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
    # Complete all checks before fitting any model or writing any output.
    checked = preflight()
    b_rows = _typed_rows(_read_csv(B_DATASET), TEAM_FEATURES)
    c_rows = _typed_rows(_read_csv(C_DATASET), tuple(V2_FEATURES))
    cutoff = DEFAULT_CUTOFF.isoformat()
    b_train = [row for row in b_rows if row["match_date"] < cutoff]
    b_test = [row for row in b_rows if row["match_date"] >= cutoff]
    c_train = [row for row in c_rows if row["match_date"] < cutoff]
    c_test = [row for row in c_rows if row["match_date"] >= cutoff]

    # Rolling-origin validation. Each fold gets independent model and preprocessor instances.
    fold_results: dict[str, Any] = {}
    fold_predictions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    fold_manifests: dict[str, Any] = {}
    fold_preprocessing: dict[str, Any] = {}
    for fold in ROLLING_ORIGIN_FOLDS:
        b_fold_train, b_validation = split_rolling_fold(b_rows, fold)
        c_prepared = prepare_rolling_fold(c_rows, fold, FoldPreprocessor(tuple(V2_FEATURES)))
        c_fold_train, c_validation = c_prepared.training_rows, c_prepared.validation_rows
        if _ordered_ids(b_fold_train) != _ordered_ids(c_fold_train) or _ordered_ids(b_validation) != _ordered_ids(c_validation):
            raise ValueError(f"B and C IDs differ in {fold.name}")
        b_model = PoissonBaseline.fit(b_fold_train)
        b_predictions = _b_predict(b_validation, b_model)
        c_preprocessor, c_home, c_away = _fit_c_model(
            c_fold_train, preprocessor=c_prepared.preprocessor,
            training_matrix=c_prepared.training_matrix,
        )
        c_predictions = _c_predict(
            c_validation, c_preprocessor, c_home, c_away,
            matrix=c_prepared.validation_matrix,
        )
        fold_results[fold.name] = {"B": evaluate_prediction_rows(b_predictions), "C": evaluate_prediction_rows(c_predictions)}
        fold_predictions[fold.name] = {"B": b_predictions, "C": c_predictions}
        fold_manifests[fold.name] = _fold_manifest(b_fold_train, b_validation)
        fold_preprocessing[fold.name] = {
            "B": {"method": "No external transformer; v1's train-fitted league goal means and shrinkage fallback are retained."},
            "C": {"fit_partition_ids_sha256": _canonical_hash(_ordered_ids(c_fold_train)), "feature_order": list(V2_FEATURES), "state": c_preprocessor.fitted_state(), "fit_warnings": getattr(c_preprocessor, "fit_warnings", [])},
        }

    # Fit final candidates only on the approved training IDs and evaluate the locked test once.
    b_model = PoissonBaseline.fit(b_train)
    b_final_predictions = _b_predict(b_test, b_model)
    c_preprocessor, c_home, c_away = _fit_c_model(c_train)
    c_final_predictions = _c_predict(c_test, c_preprocessor, c_home, c_away)
    b_final_metrics = evaluate_prediction_rows(b_final_predictions)
    c_final_metrics = evaluate_prediction_rows(c_final_predictions)
    if _ordered_ids(b_test) != _ordered_ids(c_test) or _ordered_ids(b_test) != [int(r["provider_match_id"]) for r in _read_csv(A_PREDICTIONS)]:
        raise ValueError("Final test prediction IDs do not match A/B/C")

    a_metrics, _ = _a_saved_metrics()
    config = {
        "comparison": "A frozen Poisson v1; B original six-input Poisson; C regularized Poisson log-link on approved v2 features.",
        "cutoff_date": cutoff,
        "same_utc_date_policy": "Feature datasets batch all fixtures for a UTC date before applying results; folds keep date groups intact.",
        "B": {"estimator": "Original ml.baselines.poisson.PoissonBaseline", "features": list(TEAM_FEATURES), "prior_matches": 5.0, "score_probability_grid_max_goals": GRID_MAX_GOALS, "goal_rate_clamp": [GOAL_RATE_MIN, GOAL_RATE_MAX], "missing_values": "Original shrinkage backoff to training-fitted overall home/away goal means."},
        "C": {**POISSON_V2_SETTINGS, "features": list(V2_FEATURES), "preprocessing": "Fold-only numeric median imputation, missing indicators, StandardScaler; final preprocessing fitted only on 2,556 training rows."},
        "model_difference_justification": "C needs a log-link estimator to use 13 numeric predictors; B retains the exact original multiplicative venue-rate Poisson formula. The approved addendum specifies two regularized log-link goal models for C. Shared score grid and goal-rate clamp preserve probability construction settings.",
        "folds": checked["folds"],
        "training_id_sha256": checked["ordered_training_ids_sha256"],
        "test_id_sha256": checked["ordered_test_ids_sha256"],
        "B_dataset_sha256": checked["b_dataset_sha256"],
        "C_dataset_sha256": checked["c_dataset_sha256"],
    }
    config_hash = _canonical_hash(config)
    training_manifest = {
        "cutoff_date": cutoff,
        "training_count": len(b_train), "training_fixture_ids": _ordered_ids(b_train),
        "training_fixture_ids_sha256": _canonical_hash(_ordered_ids(b_train)),
        "test_count": len(b_test), "test_fixture_ids": _ordered_ids(b_test),
        "test_fixture_ids_sha256": _canonical_hash(_ordered_ids(b_test)),
        "folds": fold_manifests,
        "excluded_fixture_ids": sorted(EXCLUDED_FIXTURE_IDS),
    }

    b_model_dir = get_output_directory("comparator_b", "model")
    c_model_dir = get_output_directory("poisson_v2", "model")
    b_eval_dir = get_output_directory("comparator_b", "evaluation")
    c_eval_dir = get_output_directory("poisson_v2", "evaluation")

    # Persist B model and configuration.
    b_model_artifact = {
        "model_version": "expanded_history_v1",
        "model_type": "independent_team_strength_poisson",
        "dataset_sha256": checked["b_dataset_sha256"],
        "configuration_sha256": config_hash,
        "training_rows": len(b_train), "training_date_range": _date_range(b_train),
        "features_used": list(TEAM_FEATURES),
        "training_goal_means": {"home": b_model.home_goal_mean, "away": b_model.away_goal_mean},
        "model_parameters": b_model.to_dict(),
    }
    _safe_json(b_model_dir / "model.json", b_model_artifact)
    _safe_json(b_model_dir / "training_config.json", config["B"])
    _safe_json(b_model_dir / "fixture_manifest.json", training_manifest)

    # Persist C estimators and their training-only preprocessor.
    _safe_joblib(c_model_dir / "home_goals_model.joblib", c_home)
    _safe_joblib(c_model_dir / "away_goals_model.joblib", c_away)
    _safe_joblib(c_model_dir / "preprocessor.joblib", c_preprocessor)
    c_model_artifact = {
        "model_version": "v2",
        "model_type": "two_independent_regularized_poisson_log_link_models",
        "dataset_sha256": checked["c_dataset_sha256"],
        "configuration_sha256": config_hash,
        "training_rows": len(c_train), "training_date_range": _date_range(c_train),
        "features_used": list(V2_FEATURES),
        "settings": POISSON_V2_SETTINGS,
        "home_model": {"intercept": float(c_home.intercept_), "coefficients": c_home.coef_.tolist()},
        "away_model": {"intercept": float(c_away.intercept_), "coefficients": c_away.coef_.tolist()},
        "preprocessing": {"feature_names_out": c_preprocessor.get_feature_names_out().tolist(), "state": c_preprocessor.fitted_state()},
    }
    _safe_json(c_model_dir / "model_metadata.json", c_model_artifact)
    _safe_json(c_model_dir / "training_config.json", config["C"])
    _safe_json(c_model_dir / "fixture_manifest.json", training_manifest)

    # Save fold-specific validation artifacts only after all evaluations completed.
    for fold_name in fold_results:
        for label, evaluation_root in (("B", b_eval_dir), ("C", c_eval_dir)):
            fold_dir = evaluation_root / "folds" / fold_name
            _safe_json(fold_dir / "metrics.json", fold_results[fold_name][label])
            _safe_csv(fold_dir / "predictions.csv", fold_predictions[fold_name][label])
            _safe_json(fold_dir / "fixture_manifest.json", fold_manifests[fold_name])
            _safe_json(fold_dir / "preprocessing.json", fold_preprocessing[fold_name][label])

    for evaluation_root, predictions, metrics in (
        (b_eval_dir, b_final_predictions, b_final_metrics),
        (c_eval_dir, c_final_predictions, c_final_metrics),
    ):
        _safe_csv(evaluation_root / "test_predictions.csv", predictions)
        _safe_json(evaluation_root / "final_evaluation.json", metrics)
        _safe_json(evaluation_root / "fixture_manifest.json", training_manifest)
        _safe_json(evaluation_root / "training_config.json", config)
    _safe_json(c_eval_dir / "final_preprocessing.json", {"feature_order": list(V2_FEATURES), "state": c_preprocessor.fitted_state(), "fit_warnings": getattr(c_preprocessor, "fit_warnings", [])})

    comparison = {"A_frozen_v1": a_metrics, "B_expanded_history": b_final_metrics, "C_redesigned_v2": c_final_metrics, "rolling_validation": fold_results}
    report = _comparison_markdown(a_metrics, b_final_metrics, c_final_metrics, fold_results)
    for evaluation_root in (b_eval_dir, c_eval_dir):
        _safe_json(evaluation_root / "comparison.json", comparison)
        report_path = assert_unprotected_output_path(evaluation_root / "comparison_report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        _safe_json(evaluation_root / "run_manifest.json", {
            "preflight": checked, "configuration": config, "configuration_sha256": config_hash,
            "fixture_manifest": training_manifest, "artifact_sha256": {
                str(path.name): _file_hash(path)
                for path in (b_model_dir / "model.json", c_model_dir / "home_goals_model.joblib", c_model_dir / "away_goals_model.joblib", c_model_dir / "preprocessor.joblib")
            },
        })
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate isolated Poisson comparators B and v2 C")
    parser.parse_args()
    result = run()
    print(json.dumps({
        "B_test": result["B_expanded_history"],
        "C_test": result["C_redesigned_v2"],
        "folds": {key: {model: value["one_x_two"] for model, value in models.items()} for key, models in result["rolling_validation"].items()},
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

