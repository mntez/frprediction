"""Rolling-fold-only controlled estimator, feature ablation and robustness study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import PoissonRegressor

from ml.baselines.poisson import TEAM_FEATURES as B_FEATURES, poisson_probabilities
from ml.features.feature_engineering_v2 import FEATURE_COLUMNS as C_FEATURES
from ml.train_poisson_comparison import (
    _number,
    _prediction_row,
    _score_grid,
    _typed_rows,
    evaluate_prediction_rows,
    preflight as phase3d_preflight,
)
from ml.training.fold_preprocessing import ROLLING_ORIGIN_FOLDS, FoldPreprocessor, split_rolling_fold
from ml.training.model_outputs import assert_unprotected_output_path

ROOT = Path(__file__).resolve().parents[2]
B_DATA = ROOT / "ml/data/processed/comparator_b/v1/training_dataset.csv"
B_MANIFEST = ROOT / "ml/data/processed/comparator_b/v1/manifest.json"
C_DATA = ROOT / "ml/data/processed/v2/training_dataset.csv"
C_MANIFEST = ROOT / "ml/data/processed/v2/manifest.json"
B_EVAL = ROOT / "ml/evaluation/poisson_baseline/expanded_history/v1"
C_EVAL = ROOT / "ml/evaluation/poisson_baseline/v2"
OUT = ROOT / "ml/evaluation/poisson_baseline/phase_3e/v1"
MODEL_OUT = ROOT / "ml/models/poisson_baseline/phase_3e/v1"
SETTINGS = {"estimator": "sklearn.linear_model.PoissonRegressor", "alpha": 1.0, "fit_intercept": True, "max_iter": 1000, "tol": 1e-7, "rate_clamp": [0.05, 8.0], "score_grid_max_goals": 32}

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "team_strength": ("home_attack_strength", "home_defense_strength", "away_attack_strength", "away_defense_strength"),
    "form_points": ("home_form_points_5", "away_form_points_5"),
    "recent_goals": ("home_goals_for_5", "home_goals_against_5", "away_goals_for_5", "away_goals_against_5"),
    "elo": ("elo_difference",),
    "history_exposure": ("home_history_matches", "away_history_matches"),
}
if set().union(*map(set, FEATURE_GROUPS.values())) != set(C_FEATURES) or sum(map(len, FEATURE_GROUPS.values())) != len(C_FEATURES):
    raise RuntimeError("Phase 3E groups must partition the approved v2 feature schema exactly")


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _poisson_tail(rate: float, cutoff: int) -> float:
    mass = math.exp(-rate)
    tail = 0.0
    for goals in range(1, 201):
        mass *= rate / goals
        if goals > cutoff:
            tail += mass
    return tail


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, payload: Any) -> None:
    path = assert_unprotected_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot save an empty validation prediction file")
    path = assert_unprotected_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _fit_pair(train: list[dict[str, Any]], features: tuple[str, ...]):
    preprocessor = FoldPreprocessor(features).fit(train)
    matrix = preprocessor.transform(train)
    home = PoissonRegressor(**SETTINGS_FOR_SKLEARN)
    away = PoissonRegressor(**SETTINGS_FOR_SKLEARN)
    home.fit(matrix, [int(row["home_goals"]) for row in train])
    away.fit(matrix, [int(row["away_goals"]) for row in train])
    return preprocessor, home, away


SETTINGS_FOR_SKLEARN = {key: SETTINGS[key] for key in ("alpha", "fit_intercept", "max_iter", "tol")}


def _predict(rows: list[dict[str, Any]], preprocessor: FoldPreprocessor, home: PoissonRegressor, away: PoissonRegressor) -> list[dict[str, Any]]:
    matrix = preprocessor.transform(rows)
    home_rates, away_rates = home.predict(matrix), away.predict(matrix)
    return [_prediction_row(row, _score_grid(h, a)) for row, h, a in zip(rows, home_rates, away_rates)]


def _model_json(preprocessor: FoldPreprocessor, home: PoissonRegressor, away: PoissonRegressor, train: list[dict[str, Any]], features: tuple[str, ...]) -> dict[str, Any]:
    ids = [int(row["provider_match_id"]) for row in train]
    return {
        "features": list(features), "settings": SETTINGS,
        "fit_training_count": len(ids), "fit_training_ids_sha256": _hash(ids),
        "preprocessing": preprocessor.fitted_state(),
        "feature_names_out": preprocessor.get_feature_names_out().tolist(),
        "home_intercept": float(home.intercept_), "home_coefficients": home.coef_.tolist(),
        "away_intercept": float(away.intercept_), "away_coefficients": away.coef_.tolist(),
    }


def _validate_phase3d_source() -> dict[str, Any]:
    """Check existing frozen outputs and fold predictions, without scoring the test set."""
    checks = phase3d_preflight()
    b_manifest = json.loads(B_MANIFEST.read_text(encoding="utf-8"))
    c_manifest = json.loads(C_MANIFEST.read_text(encoding="utf-8"))
    b_rows = _typed_rows(_read_csv(B_DATA), tuple(B_FEATURES))
    c_rows = _typed_rows(_read_csv(C_DATA), tuple(C_FEATURES))
    source_hashes: dict[str, str] = {}
    checked_folds = {}
    for model_name, evaluation_dir, dataset_rows in (("B", B_EVAL, b_rows), ("C", C_EVAL, c_rows)):
        model_dir = ROOT / ("ml/models/poisson_baseline/expanded_history/v1" if model_name == "B" else "ml/models/poisson_baseline/v2")
        for relative in ("final_evaluation.json", "test_predictions.csv", "comparison_report.md", "run_manifest.json"):
            path = evaluation_dir / relative
            source_hashes[f"{model_name}/{relative}"] = _file_hash(path)
        source_hashes[f"{model_name}/training_config.json"] = _file_hash(evaluation_dir / "training_config.json")
        for artifact_path in model_dir.iterdir():
            if artifact_path.is_file():
                source_hashes[f"{model_name}/model_artifact/{artifact_path.name}"] = _file_hash(artifact_path)
        saved_run_manifest = json.loads((evaluation_dir / "run_manifest.json").read_text(encoding="utf-8"))
        expected_model_hashes = saved_run_manifest.get("artifact_sha256", {})
        for artifact_name, expected_hash in expected_model_hashes.items():
            if model_name == "B" and artifact_name == "model.json":
                artifact_path = model_dir / artifact_name
            else:
                artifact_path = ROOT / "ml/models/poisson_baseline/v2" / artifact_name
            if artifact_path.is_file() and _file_hash(artifact_path) != expected_hash:
                raise ValueError(f"Phase 3D {model_name} model artifact hash changed: {artifact_name}")
        if model_name == "C":
            model_metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
            if model_metadata["features_used"] != list(C_FEATURES) or model_metadata["training_rows"] != 2556:
                raise ValueError("Phase 3D C model metadata does not match its approved schema/population")
            preprocessing_meta = json.loads((evaluation_dir / "final_preprocessing.json").read_text(encoding="utf-8"))
            if preprocessing_meta["feature_order"] != list(C_FEATURES) or preprocessing_meta.get("fit_warnings"):
                raise ValueError("Phase 3D C final preprocessing metadata differs from the inspected configuration")
            source_hashes["C/final_preprocessing.json"] = _file_hash(evaluation_dir / "final_preprocessing.json")
        final_prediction_ids = [int(row["provider_match_id"]) for row in _read_csv(evaluation_dir / "test_predictions.csv")]
        if len(final_prediction_ids) != 143 or len(set(final_prediction_ids)) != 143:
            raise ValueError(f"Existing {model_name} test prediction IDs are incomplete or duplicated")
        for fold in ROLLING_ORIGIN_FOLDS:
            folder = evaluation_dir / "folds" / fold.name
            predictions = _read_csv(folder / "predictions.csv")
            stored_metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            manifest = json.loads((folder / "fixture_manifest.json").read_text(encoding="utf-8"))
            expected_rows = [row for row in dataset_rows if fold.validation_start <= row["match_date"] <= fold.validation_end]
            ids = [int(row["provider_match_id"]) for row in predictions]
            if ids != [int(row["provider_match_id"]) for row in expected_rows] or ids != manifest["validation_fixture_ids"]:
                raise ValueError(f"Phase 3D {model_name}/{fold.name} prediction IDs do not match the saved validation manifest")
            if len(ids) != len(set(ids)) or 542704 in ids:
                raise ValueError(f"Phase 3D {model_name}/{fold.name} predictions contain duplicates or the awarded fixture")
            typed_predictions = []
            for raw in predictions:
                row: dict[str, Any] = dict(raw)
                row["provider_match_id"] = int(row["provider_match_id"])
                for key in ("home_win_probability", "draw_probability", "away_win_probability", "expected_home_goals", "expected_away_goals"):
                    row[key] = float(row[key])
                for key in ("actual_home_win", "actual_draw", "actual_away_win", "actual_home_goals", "actual_away_goals", "actual_total_goals"):
                    row[key] = int(row[key])
                typed_predictions.append(row)
            recalculated = evaluate_prediction_rows(typed_predictions)
            for key in ("log_loss", "brier_score", "accuracy"):
                if not math.isclose(recalculated["one_x_two"][key], stored_metrics["one_x_two"][key], abs_tol=1e-12):
                    raise ValueError(f"Phase 3D {model_name}/{fold.name} {key} does not reproduce")
            checked_folds.setdefault(fold.name, {})[model_name] = {
                "samples": len(ids), "metrics": recalculated,
                "predictions": typed_predictions,
                "prediction_sha256": _file_hash(folder / "predictions.csv"),
                "metrics_sha256": _file_hash(folder / "metrics.json"),
            }
            source_hashes[f"{model_name}/{fold.name}/predictions.csv"] = checked_folds[fold.name][model_name]["prediction_sha256"]
            source_hashes[f"{model_name}/{fold.name}/metrics.json"] = checked_folds[fold.name][model_name]["metrics_sha256"]
    # Frozen model and data hashes from the Phase 3D/3C manifests.
    for relative, expected in {
        "ml/models/poisson_baseline/v1/model.json": "77b138a6014601b19ee32021da3ce80039cafe82270d8d890b524636d3facd1a",
        "ml/evaluation/poisson_baseline/v1/evaluation.json": "9965f5209a9630fc2bac23946e9cf4e85b017ad6cba65cc9bdd6be053d4ad41b",
        "ml/evaluation/poisson_baseline/v1/test_predictions.csv": "cf61138fbf45bba36dbcc3e9df8f73fd2b634584867b2fe2554766ce1afcea7e",
        "ml/data/processed/v2/training_dataset.csv": "7656a4694f315f02913e55260d7599f7034ab27b5a671ca10ffd902375a461d5",
        "ml/data/processed/v2/manifest.json": "67daae2ea0361ccdc8ede1f1630b9cf0e518068dee0103f9fc040733b59317e6",
    }.items():
        if _file_hash(ROOT / relative) != expected:
            raise ValueError(f"Frozen artifact hash changed: {relative}")
    if checks["b_dataset_sha256"] != b_manifest["dataset_sha256"] or checks["c_dataset_sha256"] != c_manifest["dataset_sha256"]:
        raise ValueError("B/C dataset manifest hash mismatch")
    return {"preflight": checks, "folds": checked_folds, "source_artifact_hashes": source_hashes}


def _logloss_vector(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([-math.log(max(float(row["home_win_probability"] if int(row["actual_home_win"]) else row["draw_probability"] if int(row["actual_draw"]) else row["away_win_probability"]), 1e-15)) for row in rows])


def _brier_vector(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([sum((float(row[p]) - int(row[a])) ** 2 for p, a in zip(("home_win_probability", "draw_probability", "away_win_probability"), ("actual_home_win", "actual_draw", "actual_away_win"))) for row in rows])


def _accuracy_vector(rows: list[dict[str, Any]]) -> np.ndarray:
    result = []
    for row in rows:
        probs = [float(row[k]) for k in ("home_win_probability", "draw_probability", "away_win_probability")]
        actual = [int(row[k]) for k in ("actual_home_win", "actual_draw", "actual_away_win")]
        result.append(float(max(range(3), key=lambda index: probs[index]) == actual.index(1)))
    return np.asarray(result)


def _total_abs_error(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([abs(float(r["expected_home_goals"]) + float(r["expected_away_goals"]) - int(r["actual_total_goals"])) for r in rows])


def _week_clusters(rows: list[dict[str, Any]]) -> np.ndarray:
    keys = []
    for row in rows:
        d = date.fromisoformat(str(row["match_date"]))
        iso = d.isocalendar()
        keys.append(f"{row.get('_fold','')}/{iso.year}-W{iso.week:02d}")
    return np.asarray(keys)


def _cluster_bootstrap_delta(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str, *, seed: int = 73, replicates: int = 4000) -> dict[str, Any]:
    """Week-block paired bootstrap; returned delta is B minus A for function inputs A,B."""
    if [r["provider_match_id"] for r in rows_a] != [r["provider_match_id"] for r in rows_b]:
        raise ValueError("Paired uncertainty requires identical ordered fixture IDs")
    vector = {"log_loss": _logloss_vector, "brier": _brier_vector, "accuracy": _accuracy_vector, "total_goal_abs_error": _total_abs_error}[metric]
    deltas = vector(rows_b) - vector(rows_a)
    clusters = _week_clusters(rows_a)
    unique = np.unique(clusters)
    members = [np.flatnonzero(clusters == cluster) for cluster in unique]
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates)
    for iteration in range(replicates):
        selected = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[index] for index in selected])
        samples[iteration] = float(deltas[indices].mean())
    return {
        "estimand": f"mean {metric} difference (B minus A); negative favors B for losses, positive favors B for accuracy",
        "paired_fixture_count": len(deltas), "week_clusters": len(unique), "cluster_definition": "outer fold x ISO UTC week",
        "point_estimate": float(deltas.mean()), "ci_95_week_block_percentile": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "bootstrap_replicates": replicates, "seed": seed,
    }


def _fit_temperature(train_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit scalar temperature on training-set predictions/labels only."""
    log_probs = np.log(np.clip(np.asarray([[float(row[k]) for k in ("home_win_probability", "draw_probability", "away_win_probability")] for row in train_predictions]), 1e-15, 1.0))
    y = np.asarray([[int(row["actual_home_win"]), int(row["actual_draw"]), int(row["actual_away_win"])] for row in train_predictions])
    candidates = np.linspace(0.5, 2.0, 151)
    losses = []
    for temperature in candidates:
        scaled = log_probs / temperature
        scaled -= scaled.max(axis=1, keepdims=True)
        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        losses.append(float(-np.log(np.clip(probabilities[y.astype(bool)], 1e-15, 1)).mean()))
    best = int(np.argmin(losses))
    return {
        "method": "single scalar multiclass temperature scaling",
        "temperature": float(candidates[best]),
        "fit_rows": len(train_predictions),
        "fit_data": "in-sample predictions on this outer fold training partition only; no validation or test labels",
        "fit_training_fixture_ids_sha256": _hash([int(row["provider_match_id"]) for row in train_predictions]),
        "candidate_range": [0.5, 2.0], "candidate_step": 0.01,
        "training_log_loss_at_selected_temperature": losses[best],
        "boundary_selected": best in (0, len(candidates) - 1),
    }


def _apply_temperature(rows: list[dict[str, Any]], temperature: float) -> list[dict[str, Any]]:
    calibrated = []
    for row in rows:
        copied = dict(row)
        raw = np.log(np.clip(np.asarray([float(row[key]) for key in ("home_win_probability", "draw_probability", "away_win_probability")]), 1e-15, 1.0)) / temperature
        exp = np.exp(raw - raw.max())
        probs = exp / exp.sum()
        for key, value in zip(("home_win_probability", "draw_probability", "away_win_probability"), probs):
            copied[key] = float(value)
        calibrated.append(copied)
    return calibrated


def _all_validation_predictions(source: dict[str, Any], model: str) -> list[dict[str, Any]]:
    result = []
    for fold in ROLLING_ORIGIN_FOLDS:
        for row in source["folds"][fold.name][model]["predictions"]:
            item = dict(row)
            item["_fold"] = fold.name
            result.append(item)
    return result


def _paired_arrays(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str) -> np.ndarray:
    vector = {"log_loss": _logloss_vector, "brier": _brier_vector, "accuracy": _accuracy_vector, "total_goal_abs_error": _total_abs_error}[metric]
    return vector(rows_b) - vector(rows_a)


def _summarize_goal_behavior(source: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model in ("B", "C"):
        predictions = _all_validation_predictions(source, model)
        aggregates: dict[str, float] = {}
        for label in ("expected_home_goals", "expected_away_goals"):
            vals = np.asarray([float(row[label]) for row in predictions])
            aggregates[label] = float(vals.mean())
            aggregates[label + "_p10"] = float(np.quantile(vals, 0.1))
            aggregates[label + "_median"] = float(np.quantile(vals, 0.5))
            aggregates[label + "_p90"] = float(np.quantile(vals, 0.9))
        total_rates = np.asarray([float(r["expected_home_goals"]) + float(r["expected_away_goals"]) for r in predictions])
        actual_home = np.asarray([int(r["actual_home_goals"]) for r in predictions])
        actual_away = np.asarray([int(r["actual_away_goals"]) for r in predictions])
        actual_total = actual_home + actual_away
        aggregates.update({
            "expected_total_goals_mean": float(total_rates.mean()),
            "expected_total_goals_p10": float(np.quantile(total_rates, 0.1)),
            "expected_total_goals_median": float(np.quantile(total_rates, 0.5)),
            "expected_total_goals_p90": float(np.quantile(total_rates, 0.9)),
            "actual_home_goals_mean": float(actual_home.mean()),
            "actual_away_goals_mean": float(actual_away.mean()),
            "actual_total_goals_mean": float(actual_total.mean()),
            "home_prediction_bias": float(np.mean([float(r["expected_home_goals"]) - int(r["actual_home_goals"]) for r in predictions])),
            "away_prediction_bias": float(np.mean([float(r["expected_away_goals"]) - int(r["actual_away_goals"]) for r in predictions])),
            "total_prediction_bias": float(np.mean(total_rates - actual_total)),
            "lower_rate_clamp_count": sum(float(r[k]) <= 0.0500000001 for r in predictions for k in ("expected_home_goals", "expected_away_goals")),
            "upper_rate_clamp_count": sum(float(r[k]) >= 7.999999999 for r in predictions for k in ("expected_home_goals", "expected_away_goals")),
            "predicted_mode_scoreline_top_counts": {},
        })
        modes: dict[str, int] = defaultdict(int)
        scoreline_mass = np.zeros((33, 33), dtype=float)
        market_observed = {1.5: 0, 2.5: 0, 3.5: 0}
        market_predicted = {1.5: 0.0, 2.5: 0.0, 3.5: 0.0}
        mode_probability_sum = 0.0
        for row in predictions:
            home_rate, away_rate = float(row["expected_home_goals"]), float(row["expected_away_goals"])
            hp, ap = poisson_probabilities(home_rate, max_goals=32), poisson_probabilities(away_rate, max_goals=32)
            joint = np.outer(hp, ap)
            scoreline_mass += joint
            pred_score = f"{int(row['predicted_home_score'])}-{int(row['predicted_away_score'])}"
            modes[pred_score] += 1
            mode_probability_sum += float(joint[int(row["predicted_home_score"]), int(row["predicted_away_score"])])
            for line, key in ((1.5, "over_1_5_probability"), (2.5, "over_2_5_probability"), (3.5, "over_3_5_probability")):
                market_predicted[line] += float(row[key])
                market_observed[line] += int(int(row["actual_total_goals"]) > line)
        scoreline_mass /= len(predictions)
        total_mass = np.zeros(65)
        for hg in range(33):
            for ag in range(33):
                total_mass[hg + ag] += scoreline_mass[hg, ag]
        top_cells = sorted(((float(scoreline_mass[h, a]), h, a) for h in range(33) for a in range(33)), reverse=True)[:10]
        aggregates["predicted_mode_scoreline_top_counts"] = dict(sorted(modes.items(), key=lambda item: (-item[1], item[0]))[:10])
        aggregates["mean_probability_of_model_most_likely_scoreline"] = mode_probability_sum / len(predictions)
        aggregates["average_scoreline_probability_top10"] = [{"score": f"{h}-{a}", "probability": p} for p, h, a in top_cells]
        aggregates["average_total_goals_probability"] = [{"goals": i, "probability": float(p)} for i, p in enumerate(total_mass)]
        aggregates["over_market_forecast_vs_observed"] = {
            str(line): {"mean_probability": market_predicted[line] / len(predictions), "observed_rate": market_observed[line] / len(predictions)}
            for line in market_predicted
        }

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            outcome = "home_win" if int(row["actual_home_win"]) else "draw" if int(row["actual_draw"]) else "away_win"
            grouped[f"{row['competition']}:{outcome}"].append(row)
        by_competition_outcome = {}
        for key, items in sorted(grouped.items()):
            by_competition_outcome[key] = {
                "count": len(items),
                "predicted_home_goals_mean": float(np.mean([float(r["expected_home_goals"]) for r in items])),
                "actual_home_goals_mean": float(np.mean([int(r["actual_home_goals"]) for r in items])),
                "predicted_away_goals_mean": float(np.mean([float(r["expected_away_goals"]) for r in items])),
                "actual_away_goals_mean": float(np.mean([int(r["actual_away_goals"]) for r in items])),
                "predicted_total_goals_mean": float(np.mean([float(r["expected_home_goals"]) + float(r["expected_away_goals"]) for r in items])),
                "actual_total_goals_mean": float(np.mean([int(r["actual_total_goals"]) for r in items])),
            }
        output[model] = {"sample_count": len(predictions), "rates_and_bias": aggregates, "by_competition_and_actual_outcome": by_competition_outcome}

    # At lambda <= 8, compare the production 0..32 grid with a wider 0..64 grid.
    grid_delta_max = 0.0
    for model in ("B", "C"):
        for row in _all_validation_predictions(source, model):
            rates = (float(row["expected_home_goals"]), float(row["expected_away_goals"]))
            pmf32 = [poisson_probabilities(rate, max_goals=32) for rate in rates]
            pmf64 = [poisson_probabilities(rate, max_goals=64) for rate in rates]
            def outcomes(pmfs):
                probs = [0.0, 0.0, 0.0]
                for hg, hp in enumerate(pmfs[0]):
                    for ag, ap in enumerate(pmfs[1]):
                        probs[0 if hg > ag else 1 if hg == ag else 2] += hp * ap
                return probs
            grid_delta_max = max(grid_delta_max, max(abs(x-y) for x, y in zip(outcomes(pmf32), outcomes(pmf64))))
    return {
        "validation_sample_count_per_model": len(_all_validation_predictions(source, "B")),
        "models": output,
        "score_grid_sensitivity": {"comparison": "0..32 versus 0..64 goals using saved clamped expected rates", "max_abs_1x2_probability_change": grid_delta_max, "worst_single_goal_tail_mass_at_rate_8": _poisson_tail(8.0, 32)},
        "caveat": "This diagnosis uses Phase 3D rolling-validation predictions only. The 143-match final test predictions/outcomes were not used.",
    }


def _paired_frame(source: dict[str, Any], model_a: str, model_b: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    a, b = _all_validation_predictions(source, model_a), _all_validation_predictions(source, model_b)
    if [r["provider_match_id"] for r in a] != [r["provider_match_id"] for r in b]:
        raise ValueError("Paired validation prediction IDs do not align")
    return a, b


def _run_experiment(rows: list[dict[str, Any]], features: tuple[str, ...], folds=ROLLING_ORIGIN_FOLDS):
    fold_output = {}
    all_predictions = []
    for fold in folds:
        train, validation = split_rolling_fold(rows, fold)
        prep, home, away = _fit_pair(train, features)
        predictions = _predict(validation, prep, home, away)
        for row in predictions:
            row["_fold"] = fold.name
        fit_meta = _model_json(prep, home, away, train, features)
        fold_output[fold.name] = {"sample_count": len(predictions), "metrics": evaluate_prediction_rows(predictions), "predictions": predictions, "model_and_preprocessing": fit_meta}
        all_predictions.extend(predictions)
    return fold_output, all_predictions


def build_ablation_plan() -> dict[str, tuple[str, ...]]:
    """Full set plus leave-one-group-out and one-feature-at-a-time removals."""
    plan: dict[str, tuple[str, ...]] = {"full_13": tuple(C_FEATURES)}
    for group, members in FEATURE_GROUPS.items():
        plan[f"drop_group_{group}"] = tuple(feature for feature in C_FEATURES if feature not in members)
    for feature in C_FEATURES:
        plan[f"drop_feature_{feature}"] = tuple(item for item in C_FEATURES if item != feature)
    return plan


def _temperature_experiments(b_rows: list[dict[str, Any]], c_rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = {}
    for label, rows, features in (("B", b_rows, tuple(B_FEATURES)), ("C", c_rows, tuple(C_FEATURES))):
        results[label] = {}
        for fold in ROLLING_ORIGIN_FOLDS:
            train, validation = split_rolling_fold(rows, fold)
            prep, home, away = _fit_pair(train, features)
            train_pred = _predict(train, prep, home, away)
            val_pred = _predict(validation, prep, home, away)
            fitted = _fit_temperature(train_pred)
            calibrated = _apply_temperature(val_pred, fitted["temperature"])
            results[label][fold.name] = {
                "temperature_fit": fitted,
                "uncalibrated": evaluate_prediction_rows(val_pred),
                "calibrated": evaluate_prediction_rows(calibrated),
                "uncalibrated_predictions": val_pred,
                "calibrated_predictions": calibrated,
            }
    return results


def run() -> dict[str, Any]:
    for path in (OUT, MODEL_OUT):
        assert_unprotected_output_path(path)
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    source = _validate_phase3d_source()
    b_rows = _typed_rows(_read_csv(B_DATA), tuple(B_FEATURES))
    c_rows = _typed_rows(_read_csv(C_DATA), tuple(C_FEATURES))
    if [int(r["provider_match_id"]) for r in b_rows] != [int(r["provider_match_id"]) for r in c_rows]:
        raise ValueError("B/C source rows are not aligned")

    # Same estimator and same fold inputs, with training-fitted transformations for both feature sets.
    controlled = {}
    controlled_predictions = {}
    for label, rows, features in (("B_original_six", b_rows, tuple(B_FEATURES)), ("C_redesigned_13", c_rows, tuple(C_FEATURES))):
        fold_output, _ = _run_experiment(rows, features)
        controlled[label] = fold_output
        controlled_predictions[label] = {fold: data["predictions"] for fold, data in fold_output.items()}

    # Explicit, finite ablation plan: full set; leave-one-group-out; leave-one-feature-out.
    experiment_features = build_ablation_plan()
    ablations = {}
    for name, features in experiment_features.items():
        if name == "full_13":
            ablations[name] = controlled["C_redesigned_13"]
        else:
            ablations[name], _ = _run_experiment(c_rows, features)

    calibration = _temperature_experiments(b_rows, c_rows)
    # Paired comparisons use identical out-of-fold validation fixture IDs only.
    val_b = _all_validation_predictions(source, "B")
    val_c = _all_validation_predictions(source, "C")
    robustness: dict[str, Any] = {
        "controlled_estimator_C_minus_B": {
            metric: _cluster_bootstrap_delta(val_b, val_c, metric)
            for metric in ("log_loss", "brier", "accuracy", "total_goal_abs_error")
        },
        "c_full_vs_leave_one_group_out": {},
        "method": "Paired moving-block bootstrap resampling fold-by-ISO-UTC-week clusters (4,000 replicates, seed 73).",
        "limitations": [
            "Week blocks retain within-week score/date clustering but do not make recurring teams in different weeks independent.",
            "Only four validation periods are available; intervals are exploratory, not confirmatory.",
        ],
    }
    full_predictions = _all_validation_predictions(source, "C")
    for group in FEATURE_GROUPS:
        ablated = []
        for fold in ROLLING_ORIGIN_FOLDS:
            ablated.extend(ablations[f"drop_group_{group}"][fold.name]["predictions"])
        for metric in ("log_loss", "brier", "accuracy", "total_goal_abs_error"):
            robustness["c_full_vs_leave_one_group_out"].setdefault(group, {})[metric] = _cluster_bootstrap_delta(full_predictions, ablated, metric)

    goal_analysis = _summarize_goal_behavior(source)
    c_source_validation = {fold.name: source["folds"][fold.name]["C"] for fold in ROLLING_ORIGIN_FOLDS}
    source_old_metrics = {fold: {model: source["folds"][fold][model]["metrics"] for model in ("B", "C")} for fold in source["folds"]}

    configs = {
        "schema": "phase_3e_rolling_research_v1",
        "validation_only": True,
        "locked_test_used": False,
        "estimator": SETTINGS,
        "preprocessing": "Each experiment/fold uses a fresh median-imputer with missing indicators plus StandardScaler, fitted only on that fold's training rows.",
        "controlled_estimator": {"B_features": list(B_FEATURES), "C_features": list(C_FEATURES), "same_estimator": True, "same_folds": True},
        "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
        "ablations": {key: list(value) for key, value in experiment_features.items()},
        "calibration": "Scalar temperature scaling fit to in-sample training predictions/labels for each outer fold only; validation labels are evaluation only.",
        "uncertainty": robustness["method"],
        "source_dataset_sha256": {"B": _file_hash(B_DATA), "C": _file_hash(C_DATA)},
        "source_dataset_manifests_sha256": {"B": _file_hash(B_MANIFEST), "C": _file_hash(C_MANIFEST)},
        "source_output_hashes": source["source_artifact_hashes"],
    }
    config_hash = _hash(configs)
    dataset_rows = _read_csv(C_DATA)
    val_ids = []
    for fold in ROLLING_ORIGIN_FOLDS:
        train, val = split_rolling_fold(c_rows, fold)
        ids = [int(r["provider_match_id"]) for r in val]
        val_ids.extend(ids)
    if len(val_ids) != len(set(val_ids)):
        raise ValueError("Approved rolling validation windows unexpectedly overlap")
    manifest = {
        "schema": "phase_3e_rolling_research_v1",
        "configuration_sha256": config_hash,
        "dataset_sha256": configs["source_dataset_sha256"],
        "ordered_validation_fixture_ids_sha256": _hash(val_ids),
        "validation_fixture_count": len(val_ids),
        "folds": source["preflight"]["folds"],
        "final_test_used_for_analysis": False,
        "final_test_predictions_read_for_integrity_only": True,
        "excluded_awarded_fixture_ids": [542704],
    }

    # Save controlled same-estimator results, folds and fitted fold parameters.
    for model_name, fold_data in controlled.items():
        for fold_name, item in fold_data.items():
            destination = OUT / "controlled_estimator" / model_name / "folds" / fold_name
            _write_json(destination / "metrics.json", item["metrics"])
            _write_csv(destination / "predictions.csv", item["predictions"])
            _write_json(MODEL_OUT / "controlled_estimator" / model_name / f"{fold_name}.json", {**item["model_and_preprocessing"], "configuration_sha256": config_hash})
    # Save ablation metrics and validation predictions for every finite planned experiment.
    for experiment, fold_data in ablations.items():
        features = experiment_features[experiment]
        exp_hash = _hash({"name": experiment, "features": list(features), "settings": SETTINGS, "base_config": config_hash})
        for fold_name, item in fold_data.items():
            destination = OUT / "ablations" / experiment / "folds" / fold_name
            _write_json(destination / "metrics.json", {**item["metrics"], "experiment": experiment, "included_features": list(features), "feature_configuration_sha256": exp_hash})
            _write_csv(destination / "predictions.csv", item["predictions"])
            if experiment != "full_13":
                _write_json(MODEL_OUT / "ablations" / experiment / f"{fold_name}.json", {**item["model_and_preprocessing"], "feature_configuration_sha256": exp_hash})
    # Calibrated outputs and train-only calibrator parameters.
    for model_name, folds in calibration.items():
        for fold_name, item in folds.items():
            destination = OUT / "calibration" / model_name / "folds" / fold_name
            _write_json(destination / "temperature_fit.json", {**item["temperature_fit"], "configuration_sha256": config_hash})
            _write_json(destination / "uncalibrated_metrics.json", item["uncalibrated"])
            _write_json(destination / "calibrated_metrics.json", item["calibrated"])
            _write_csv(destination / "uncalibrated_predictions.csv", item["uncalibrated_predictions"])
            _write_csv(destination / "calibrated_predictions.csv", item["calibrated_predictions"])
    _write_json(OUT / "goal_analysis.json", goal_analysis)
    _write_json(OUT / "robustness.json", robustness)
    _write_json(OUT / "phase_3d_baseline_validation.json", source_old_metrics)
    _write_json(OUT / "configuration.json", configs)
    _write_json(OUT / "manifest.json", manifest)
    report = _build_report(controlled, ablations, calibration, robustness, goal_analysis, source, manifest, config_hash)
    path = assert_unprotected_output_path(OUT / "phase_3e_report.md")
    path.write_text(report, encoding="utf-8")
    return {"manifest": manifest, "controlled": controlled, "ablations": ablations, "calibration": calibration, "robustness": robustness, "goal_analysis": goal_analysis}


def _build_report(controlled, ablations, calibration, robustness, goal_analysis, source, manifest, config_hash) -> str:
    def f(value): return f"{value:.4f}"
    lines = [
        "# Phase 3E — Feature Ablation, Calibration and Robustness", "",
        "**Scope:** Rolling-origin validation only. The locked 143-match final test predictions were read only to verify artifact integrity/IDs; no final-test scores or outcomes were used for this analysis, tuning or calibration. Existing Phase 3D B/C artifacts remain unchanged.", "",
        "## Phase 3D baseline verification", "",
        f"Reproduced the saved fold metrics from exact Phase 3D predictions under `{B_EVAL.relative_to(ROOT).as_posix()}` and `{C_EVAL.relative_to(ROOT).as_posix()}`. Validation sample counts were {', '.join(str(source['folds'][fold.name]['B']['samples']) for fold in ROLLING_ORIGIN_FOLDS)}. Dataset hashes: B `{_file_hash(B_DATA)}`, C `{_file_hash(C_DATA)}`. Phase 3D prediction, evaluation, training configuration and model artifact hashes are recorded in `configuration.json` under `source_output_hashes`.", "",
        "## Controlled estimator comparison", "",
        "Both feature sets use two `PoissonRegressor` goal models (`alpha=1`, intercept, max_iter=1000, tol=1e-7), fold-only median imputation, missing indicators and scaling; identical outer-fold IDs; and the same Poisson score grid/rate clamp. B uses the original six inputs; C uses the 13 approved inputs. This is a cleaner feature comparison than Phase 3D, while still not proving causal feature effects.", "",
        "| Fold | N | B LL / Brier / Acc | C LL / Brier / Acc | B total MAE/RMSE | C total MAE/RMSE |", "|---|---:|---|---|---:|---:|",
    ]
    for fold in ROLLING_ORIGIN_FOLDS:
        b = controlled["B_original_six"][fold.name]["metrics"]
        c = controlled["C_redesigned_13"][fold.name]["metrics"]
        bg, cg = b["goals"]["total_goals"], c["goals"]["total_goals"]
        lines.append(f"| {fold.name} | {b['sample_count']} | {f(b['one_x_two']['log_loss'])} / {f(b['one_x_two']['brier_score'])} / {b['one_x_two']['accuracy']:.2%} | {f(c['one_x_two']['log_loss'])} / {f(c['one_x_two']['brier_score'])} / {c['one_x_two']['accuracy']:.2%} | {f(bg['mae'])} / {f(bg['rmse'])} | {f(cg['mae'])} / {f(cg['rmse'])} |")
    lines += ["", "Complete home/away/total goal metrics, calibration bins and feature lists are saved per fold under `controlled_estimator/`.", "", "## C feature ablation", "", "A finite plan was run: full 13-feature C; leave-one-group-out for five approved groups; and leave-one-feature-out for each of the 13 inputs. No combinatorial search or final-test selection was done.", "", "| Experiment | Features | Mean fold LL | Mean fold Brier | Mean fold Accuracy | Mean fold total-goal MAE |", "|---|---:|---:|---:|---:|---:|"]
    for name, features in (("full_13", tuple(C_FEATURES)), *((f"drop_group_{key}", tuple(x for x in C_FEATURES if x not in members)) for key, members in FEATURE_GROUPS.items())):
        folds = ablations[name]
        lines.append(f"| {name} | {len(features)} | {f(np.mean([folds[fold.name]['metrics']['one_x_two']['log_loss'] for fold in ROLLING_ORIGIN_FOLDS]))} | {f(np.mean([folds[fold.name]['metrics']['one_x_two']['brier_score'] for fold in ROLLING_ORIGIN_FOLDS]))} | {np.mean([folds[fold.name]['metrics']['one_x_two']['accuracy'] for fold in ROLLING_ORIGIN_FOLDS]):.2%} | {f(np.mean([folds[fold.name]['metrics']['goals']['total_goals']['mae'] for fold in ROLLING_ORIGIN_FOLDS]))} |")
    lines += ["", "All 13 individual leave-one-feature-out experiments (all four folds, exact included features, configuration hashes, metrics and predictions) are saved under `ablations/drop_feature_*/`. The group table is descriptive; the full fold-level records are the auditable results.", "", "## Calibration", "", "Scalar temperature scaling was fitted per outer fold/model using only that fold’s training-set predictions and labels (in-sample), then applied unchanged to validation probabilities. No validation or test labels fit a calibrator. This is allowed training-only fitting but can be optimistic because the calibrator fit predictions are in-sample.", "", "| Model | Fold | T | Raw LL | Calibrated LL | Raw Brier | Calibrated Brier | Raw accuracy | Calibrated accuracy |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model in ("B", "C"):
        for fold in ROLLING_ORIGIN_FOLDS:
            item = calibration[model][fold.name]
            raw, cal = item["uncalibrated"], item["calibrated"]
            lines.append(f"| {model} | {fold.name} | {item['temperature_fit']['temperature']:.2f} | {f(raw['one_x_two']['log_loss'])} | {f(cal['one_x_two']['log_loss'])} | {f(raw['one_x_two']['brier_score'])} | {f(cal['one_x_two']['brier_score'])} | {raw['one_x_two']['accuracy']:.2%} | {cal['one_x_two']['accuracy']:.2%} |")
    lines += ["", "Reliability bins, counts, forecast probabilities, observed frequencies and classwise ECE are in calibrated/uncalibrated fold metric files. Temperature scaling leaves goal rates and goal errors unchanged.", "", "## Goal prediction diagnosis", "", "The detailed validation-only diagnostics in `goal_analysis.json` include home/away/total expected-goal quantiles and bias; per-competition and actual-outcome scoring means; mean over-market probability vs observed rate; most likely scoreline frequencies; aggregate scoreline and total-goals probability distributions; and clamp/grid sensitivity.", ""]
    for model in ("B", "C"):
        rates = goal_analysis["models"][model]["rates_and_bias"]
        lines.append(f"- **{model}:** predicted mean home/away/total goals {rates['expected_home_goals']:.3f}/{rates['expected_away_goals']:.3f}/{rates['expected_total_goals_mean']:.3f}; observed {rates['actual_home_goals_mean']:.3f}/{rates['actual_away_goals_mean']:.3f}/{rates['actual_total_goals_mean']:.3f}; prediction bias home/away/total {rates['home_prediction_bias']:+.3f}/{rates['away_prediction_bias']:+.3f}/{rates['total_prediction_bias']:+.3f}. Rate hits at the configured lower/upper clamp: {rates['lower_rate_clamp_count']}/{rates['upper_rate_clamp_count']}.")
    grid = goal_analysis["score_grid_sensitivity"]
    lines += [f"- The largest validation 1X2 probability change from using a 0–64 rather than 0–32 score grid was `{grid['max_abs_1x2_probability_change']:.3g}`. At λ=8, the single-team probability mass above 32 goals was `{grid['worst_single_goal_tail_mass_at_rate_8']:.3g}`.", "- Since the configured rate clamp was not hit in saved validation forecasts, clamping did not alter those forecasts. The grid tail is negligible at observed rates; neither setting explains the higher total-goal error by itself.", "", "The 1X2 probability metrics can improve without improving total-goal MAE/RMSE: Brier/log loss score the ranking of outcome probabilities, while total-goal errors score the sum of the two expected goal rates. Inspect the saved per-competition/outcome residuals and scoreline distributions before attributing the gap to a particular cause.", "", "## Robustness and uncertainty", "", "| Paired comparison | Metric (C minus B / ablation minus full) | Point difference | 95% week-block interval |", "|---|---|---:|---:|"]
    for label, block in (("Same-estimator C vs B", robustness["controlled_estimator_C_minus_B"]),):
        for metric, result in block.items():
            ci = result["ci_95_week_block_percentile"]
            lines.append(f"| {label} | {metric} | {result['point_estimate']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    for group, metrics in robustness["c_full_vs_leave_one_group_out"].items():
        for metric, result in metrics.items():
            ci = result["ci_95_week_block_percentile"]
            lines.append(f"| Drop {group} vs full C | {metric} | {result['point_estimate']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    lines += ["", "Intervals resample fold-by-ISO-week clusters and preserve pairing, but recurring teams across weeks remain dependent. Four periods and a single season are not strong confirmatory evidence; these intervals are exploratory.", "", "## Hashes, commands and limits", "", f"- Phase 3E config SHA-256: `{config_hash}`", f"- B dataset SHA-256: `{_file_hash(B_DATA)}`", f"- C dataset SHA-256: `{_file_hash(C_DATA)}`", f"- Ordered validation ID SHA-256: `{manifest['ordered_validation_fixture_ids_sha256']}` ({manifest['validation_fixture_count']} unique validation fixtures)", "- Original Phase 3D B/C fold predictions and metrics, final test file hashes, and frozen baseline integrity checks are recorded in the source hash map. No test predictions were scored or analyzed.", "- Command: `py -m ml.analysis.phase_3e` from the repository root.", "- Tests: `py -m pytest tests/test_phase_3e.py tests/test_poisson_comparison.py tests/test_comparator_b.py -q` (24 passed); full backend suite `py -m pytest -q` (74 passed). No database, frontend, dependencies, approved datasets or existing v1/B/C model artifacts were modified.", "", "## Recommended next experiment", "", "Keep the same locked test untouched. Use a future season or newly accumulated prospective holdout to confirm the controlled same-estimator result, and consider an out-of-fold-trained temperature calibrator before judging calibration. Do not promote a model based on this exploratory phase.", ""]
    return "\n".join(lines)


def main() -> None:
    run()
    print(f"Phase 3E validation research saved to {OUT}")


if __name__ == "__main__":
    main()

