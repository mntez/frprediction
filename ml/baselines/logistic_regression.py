from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from statistics import mean, median, pstdev
from typing import Any, Protocol

from sklearn.linear_model import LogisticRegression

from ml.baselines.poisson import split_chronologically


NUMERIC_FEATURES = (
    "season_start_year",
    "home_previous_matches_played", "home_previous_wins", "home_previous_draws",
    "home_previous_losses", "home_previous_goals_scored", "home_previous_goals_conceded",
    "home_rolling_avg_goals_scored", "home_rolling_avg_goals_conceded",
    "home_home_matches_played", "home_home_win_rate", "home_home_goals_scored_avg",
    "home_home_goals_conceded_avg",
    "away_previous_matches_played", "away_previous_wins", "away_previous_draws",
    "away_previous_losses", "away_previous_goals_scored", "away_previous_goals_conceded",
    "away_rolling_avg_goals_scored", "away_rolling_avg_goals_conceded",
    "away_away_matches_played", "away_away_win_rate", "away_away_goals_scored_avg",
    "away_away_goals_conceded_avg",
    "h2h_previous_matches", "h2h_home_team_wins", "h2h_draws", "h2h_home_team_losses",
)
CATEGORICAL_FEATURES = (
    "competition", "home_team", "away_team", "home_recent_form", "away_recent_form",
)
TARGETS = ("home_win", "draw", "away_win")
MISSING_CATEGORY = "__MISSING__"
CLASS_LABELS = {0: "home_win", 1: "draw", 2: "away_win"}


class ProbabilisticClassifier(Protocol):
    classes_: Any

    def predict_proba(self, features: list[list[float]]) -> Any: ...


def target_class(row: dict[str, Any]) -> int:
    matches = [index for index, target in enumerate(TARGETS) if int(row[target]) == 1]
    if len(matches) != 1 or any(int(row[target]) not in (0, 1) for target in TARGETS):
        raise ValueError(f"Invalid 1X2 targets for match {row.get('provider_match_id')}")
    return matches[0]


@dataclass
class FittedPreprocessor:
    numeric_stats: dict[str, dict[str, float]]
    categorical_values: dict[str, list[str]]

    @classmethod
    def fit(cls, training_rows: list[dict[str, Any]]) -> "FittedPreprocessor":
        if not training_rows:
            raise ValueError("Cannot fit preprocessing without training rows")
        numeric_stats: dict[str, dict[str, float]] = {}
        for column in NUMERIC_FEATURES:
            values = []
            for row in training_rows:
                value = row.get(column)
                if value is not None and value != "":
                    numeric = float(value)
                    if isfinite(numeric):
                        values.append(numeric)
            column_median = median(values) if values else 0.0
            imputed = [
                float(row[column]) if row.get(column) not in (None, "") else column_median
                for row in training_rows
            ]
            column_mean = mean(imputed)
            column_scale = pstdev(imputed) or 1.0
            numeric_stats[column] = {
                "median": float(column_median),
                "mean": float(column_mean),
                "scale": float(column_scale),
            }

        categorical_values = {}
        for column in CATEGORICAL_FEATURES:
            observed = {
                str(row[column]) for row in training_rows
                if row.get(column) not in (None, "")
            }
            categorical_values[column] = sorted(observed | {MISSING_CATEGORY})
        return cls(numeric_stats=numeric_stats, categorical_values=categorical_values)

    @property
    def feature_names(self) -> list[str]:
        names = [f"{column}__scaled" for column in NUMERIC_FEATURES]
        names.extend(f"{column}__missing" for column in NUMERIC_FEATURES)
        for column in CATEGORICAL_FEATURES:
            names.extend(f"{column}=={value}" for value in self.categorical_values[column])
        return names

    def transform(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        if not self.numeric_stats or not self.categorical_values:
            raise ValueError("Preprocessor must be fitted before transform")
        output: list[list[float]] = []
        for row in rows:
            values: list[float] = []
            for column in NUMERIC_FEATURES:
                raw = row.get(column)
                missing = raw is None or raw == ""
                value = self.numeric_stats[column]["median"] if missing else float(raw)
                if not isfinite(value):
                    missing = True
                    value = self.numeric_stats[column]["median"]
                stats = self.numeric_stats[column]
                values.append((value - stats["mean"]) / stats["scale"])
            for column in NUMERIC_FEATURES:
                raw = row.get(column)
                values.append(float(raw is None or raw == "" or not isfinite(float(raw))))
            for column in CATEGORICAL_FEATURES:
                raw = row.get(column)
                category = str(raw) if raw not in (None, "") else MISSING_CATEGORY
                known_values = self.categorical_values[column]
                values.extend(float(category == known) for known in known_values)
            output.append(values)
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "missing_category": MISSING_CATEGORY,
            "missing_numeric_strategy": "training median with a separate missing indicator",
            "categorical_strategy": "training categories one-hot encoded; unseen test categories all-zero",
            "scaling": "training mean and population standard deviation; zero-variance scale defaults to 1",
            "numeric_stats": self.numeric_stats,
            "categorical_values": self.categorical_values,
            "transformed_feature_names": self.feature_names,
        }


def fit_classifier(training_rows: list[dict[str, Any]]) -> tuple[FittedPreprocessor, LogisticRegression]:
    preprocessor = FittedPreprocessor.fit(training_rows)
    matrix = preprocessor.transform(training_rows)
    labels = [target_class(row) for row in training_rows]
    if set(labels) != {0, 1, 2}:
        raise ValueError("Training split must contain home wins, draws, and away wins")
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=3000,
        random_state=0,
    )
    classifier.fit(matrix, labels)
    return preprocessor, classifier


def _calibration(predictions: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    samples = []
    for row in predictions:
        probabilities = [row["home_win_probability"], row["draw_probability"], row["away_win_probability"]]
        top_index = max(range(3), key=lambda index: probabilities[index])
        samples.append((max(probabilities), int(top_index == row["actual_class"])))
    bin_rows = []
    ece = 0.0
    for bucket in range(bins):
        low, high = bucket / bins, (bucket + 1) / bins
        members = [sample for sample in samples if low <= sample[0] < high or (bucket == bins - 1 and sample[0] == 1.0)]
        if not members:
            continue
        confidence = sum(item[0] for item in members) / len(members)
        accuracy = sum(item[1] for item in members) / len(members)
        ece += len(members) / len(samples) * abs(confidence - accuracy)
        bin_rows.append({
            "lower_bound": low,
            "upper_bound": high,
            "count": len(members),
            "mean_confidence": confidence,
            "accuracy": accuracy,
        })
    return {
        "top_class_expected_calibration_error": ece,
        "top_class_mean_confidence": sum(item[0] for item in samples) / len(samples),
        "top_class_accuracy": sum(item[1] for item in samples) / len(samples),
        "bins": bin_rows,
    }


def evaluate_classifier(
    rows: list[dict[str, Any]], preprocessor: FittedPreprocessor, classifier: ProbabilisticClassifier
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Cannot evaluate without test rows")
    matrix = preprocessor.transform(rows)
    probabilities_by_row = classifier.predict_proba(matrix)
    class_order = list(classifier.classes_)
    evaluated = []
    for row, class_probabilities in zip(rows, probabilities_by_row):
        probabilities = {int(label): float(class_probabilities[index]) for index, label in enumerate(class_order)}
        actual = target_class(row)
        predicted = max(range(3), key=lambda label: probabilities[label])
        evaluated.append({
            "provider_match_id": row["provider_match_id"],
            "match_date": row["match_date"],
            "competition": row["competition"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "actual_class": actual,
            "actual_label": CLASS_LABELS[actual],
            "predicted_class": predicted,
            "predicted_label": CLASS_LABELS[predicted],
            "home_win_probability": probabilities[0],
            "draw_probability": probabilities[1],
            "away_win_probability": probabilities[2],
            "actual_home_win": int(actual == 0),
            "actual_draw": int(actual == 1),
            "actual_away_win": int(actual == 2),
        })

    confusion = [[0 for _ in range(3)] for _ in range(3)]
    log_loss = brier = correct = 0.0
    for row in evaluated:
        actual = row["actual_class"]
        predicted = row["predicted_class"]
        confusion[actual][predicted] += 1
        probability = row[("home_win_probability", "draw_probability", "away_win_probability")[actual]]
        log_loss -= log(max(probability, 1e-15))
        brier += sum((row[key] - float(index == actual)) ** 2 for index, key in enumerate((
            "home_win_probability", "draw_probability", "away_win_probability"
        )))
        correct += int(actual == predicted)

    per_class = {}
    for index, label in CLASS_LABELS.items():
        true_positive = confusion[index][index]
        false_positive = sum(confusion[actual][index] for actual in range(3) if actual != index)
        false_negative = sum(confusion[index][predicted] for predicted in range(3) if predicted != index)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[index]),
        }
    metrics = {
        "log_loss": log_loss / len(evaluated),
        "brier_score": brier / len(evaluated),
        "accuracy": correct / len(evaluated),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [CLASS_LABELS[index] for index in range(3)],
            "rows_actual_columns_predicted": confusion,
        },
        "calibration": _calibration(evaluated),
    }
    return metrics, evaluated


def split_using_poisson_policy(
    rows: list[dict[str, Any]], cutoff_date
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Share the Poisson split implementation to keep both test sets identical."""
    return split_chronologically(rows, cutoff_date)
