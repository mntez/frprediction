from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import exp, isfinite, log, sqrt
from typing import Any


TARGET_COLUMNS = (
    "home_goals", "away_goals", "home_win", "draw", "away_win", "total_goals",
    "over_1_5", "over_2_5", "over_3_5", "btts",
)
TEAM_FEATURES = (
    "home_home_matches_played", "home_home_goals_scored_avg",
    "home_home_goals_conceded_avg", "away_away_matches_played",
    "away_away_goals_scored_avg", "away_away_goals_conceded_avg",
)


def split_chronologically(
    rows: list[dict[str, Any]], cutoff_date: date
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Use rows strictly before cutoff for training and cutoff onward for testing."""
    ordered = sorted(rows, key=lambda row: (str(row["match_date"]), int(row["provider_match_id"])))
    training = [row for row in ordered if date.fromisoformat(str(row["match_date"])) < cutoff_date]
    testing = [row for row in ordered if date.fromisoformat(str(row["match_date"])) >= cutoff_date]
    if not training or not testing:
        raise ValueError("Chronological cutoff must leave non-empty training and test periods")
    latest_train = max(date.fromisoformat(str(row["match_date"])) for row in training)
    earliest_test = min(date.fromisoformat(str(row["match_date"])) for row in testing)
    if latest_train >= cutoff_date or earliest_test < cutoff_date or latest_train >= earliest_test:
        raise ValueError("Chronological split overlaps or includes future training rows")
    return training, testing


def target_labels(home_goals: int, away_goals: int) -> dict[str, int]:
    total = home_goals + away_goals
    return {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_win": int(home_goals > away_goals),
        "draw": int(home_goals == away_goals),
        "away_win": int(home_goals < away_goals),
        "total_goals": total,
        "over_1_5": int(total > 1.5),
        "over_2_5": int(total > 2.5),
        "over_3_5": int(total > 3.5),
        "btts": int(home_goals > 0 and away_goals > 0),
    }


def poisson_probabilities(rate: float, *, max_goals: int = 32) -> list[float]:
    if not isfinite(rate) or rate < 0:
        raise ValueError("Poisson rate must be a finite non-negative number")
    if max_goals < 1:
        raise ValueError("max_goals must be positive")
    probabilities = [exp(-rate)]
    for goals in range(1, max_goals + 1):
        probabilities.append(probabilities[-1] * rate / goals)
    total = sum(probabilities)
    if total <= 0 or not isfinite(total):
        raise ValueError("Could not calculate a valid Poisson distribution")
    # Normalizing the tiny truncated tail makes the finite score grid a valid PMF.
    return [probability / total for probability in probabilities]


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    return result if isfinite(result) else None


def _shrink(value: Any, sample_count: Any, prior_mean: float, prior_matches: float) -> float:
    observed = _number(value)
    count = _number(sample_count) or 0.0
    if observed is None or count <= 0:
        return prior_mean
    return (observed * count + prior_mean * prior_matches) / (count + prior_matches)


@dataclass(frozen=True)
class PoissonBaseline:
    home_goal_mean: float
    away_goal_mean: float
    prior_matches: float = 5.0
    max_goals: int = 32
    minimum_goal_rate: float = 0.05
    maximum_goal_rate: float = 8.0

    @classmethod
    def fit(cls, training_rows: list[dict[str, Any]]) -> "PoissonBaseline":
        if not training_rows:
            raise ValueError("Cannot fit the Poisson baseline without training rows")
        home_mean = sum(float(row["home_goals"]) for row in training_rows) / len(training_rows)
        away_mean = sum(float(row["away_goals"]) for row in training_rows) / len(training_rows)
        if home_mean <= 0 or away_mean <= 0:
            raise ValueError("Training goal means must be positive")
        return cls(home_goal_mean=home_mean, away_goal_mean=away_mean)

    def expected_goals(self, row: dict[str, Any]) -> tuple[float, float]:
        home_attack = _shrink(
            row.get("home_home_goals_scored_avg"),
            row.get("home_home_matches_played"),
            self.home_goal_mean,
            self.prior_matches,
        )
        away_defence = _shrink(
            row.get("away_away_goals_conceded_avg"),
            row.get("away_away_matches_played"),
            self.home_goal_mean,
            self.prior_matches,
        )
        away_attack = _shrink(
            row.get("away_away_goals_scored_avg"),
            row.get("away_away_matches_played"),
            self.away_goal_mean,
            self.prior_matches,
        )
        home_defence = _shrink(
            row.get("home_home_goals_conceded_avg"),
            row.get("home_home_matches_played"),
            self.away_goal_mean,
            self.prior_matches,
        )
        home_rate = home_attack * away_defence / self.home_goal_mean
        away_rate = away_attack * home_defence / self.away_goal_mean
        return (
            min(max(home_rate, self.minimum_goal_rate), self.maximum_goal_rate),
            min(max(away_rate, self.minimum_goal_rate), self.maximum_goal_rate),
        )

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        home_rate, away_rate = self.expected_goals(row)
        home_pmf = poisson_probabilities(home_rate, max_goals=self.max_goals)
        away_pmf = poisson_probabilities(away_rate, max_goals=self.max_goals)
        home_win = draw = away_win = 0.0
        over_1_5 = over_2_5 = over_3_5 = 0.0
        most_likely = (-1.0, 0, 0)
        for home_goals, home_probability in enumerate(home_pmf):
            for away_goals, away_probability in enumerate(away_pmf):
                probability = home_probability * away_probability
                if home_goals > away_goals:
                    home_win += probability
                elif home_goals == away_goals:
                    draw += probability
                else:
                    away_win += probability
                total_goals = home_goals + away_goals
                over_1_5 += probability * int(total_goals > 1.5)
                over_2_5 += probability * int(total_goals > 2.5)
                over_3_5 += probability * int(total_goals > 3.5)
                if probability > most_likely[0]:
                    most_likely = (probability, home_goals, away_goals)
        return {
            "expected_home_goals": home_rate,
            "expected_away_goals": away_rate,
            "home_win_probability": home_win,
            "draw_probability": draw,
            "away_win_probability": away_win,
            "over_1_5_probability": over_1_5,
            "over_2_5_probability": over_2_5,
            "over_3_5_probability": over_3_5,
            "btts_probability": (1 - home_pmf[0]) * (1 - away_pmf[0]),
            "predicted_home_score": most_likely[1],
            "predicted_away_score": most_likely[2],
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _brier_binary(probabilities: list[float], actuals: list[int]) -> float:
    return sum((probability - actual) ** 2 for probability, actual in zip(probabilities, actuals)) / len(actuals)


def _calibration(predictions: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    samples: list[tuple[float, int]] = []
    for prediction in predictions:
        probabilities = [prediction["home_win_probability"], prediction["draw_probability"], prediction["away_win_probability"]]
        actual_index = 0 if prediction["actual_home_win"] else 1 if prediction["actual_draw"] else 2
        top_index = max(range(3), key=lambda index: probabilities[index])
        samples.append((max(probabilities), int(top_index == actual_index)))
    bucket_rows = []
    total_error = 0.0
    for bucket in range(bins):
        low, high = bucket / bins, (bucket + 1) / bins
        items = [item for item in samples if low <= item[0] < high or (bucket == bins - 1 and item[0] == 1.0)]
        if not items:
            continue
        confidence = sum(item[0] for item in items) / len(items)
        accuracy = sum(item[1] for item in items) / len(items)
        total_error += len(items) / len(samples) * abs(confidence - accuracy)
        bucket_rows.append({"lower_bound": low, "upper_bound": high, "count": len(items), "mean_confidence": confidence, "accuracy": accuracy})
    return {
        "top_class_expected_calibration_error": total_error,
        "top_class_mean_confidence": sum(item[0] for item in samples) / len(samples),
        "top_class_accuracy": sum(item[1] for item in samples) / len(samples),
        "bins": bucket_rows,
    }


def evaluate_predictions(rows: list[dict[str, Any]], model: PoissonBaseline) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Cannot evaluate the Poisson baseline without test rows")
    evaluated = []
    for row in rows:
        actual = target_labels(int(row["home_goals"]), int(row["away_goals"]))
        for key, value in actual.items():
            if int(row[key]) != value:
                raise ValueError(f"Dataset target {key} does not match scores for match {row['provider_match_id']}")
        predicted = model.predict(row)
        evaluated.append({
            "provider_match_id": row["provider_match_id"],
            "match_date": row["match_date"],
            "competition": row["competition"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            **predicted,
            **{f"actual_{key}": value for key, value in actual.items()},
        })

    log_loss = 0.0
    brier_score = 0.0
    correct = 0
    home_errors: list[float] = []
    away_errors: list[float] = []
    total_goal_errors: list[float] = []
    class_keys = ("home_win_probability", "draw_probability", "away_win_probability")
    for row in evaluated:
        probabilities = [row[key] for key in class_keys]
        actual = [row["actual_home_win"], row["actual_draw"], row["actual_away_win"]]
        actual_probability = max(probabilities[index] for index, outcome in enumerate(actual) if outcome)
        log_loss -= log(max(actual_probability, 1e-15))
        brier_score += sum((probability - target) ** 2 for probability, target in zip(probabilities, actual))
        correct += int(max(range(3), key=lambda index: probabilities[index]) == actual.index(1))
        home_errors.append(row["expected_home_goals"] - row["actual_home_goals"])
        away_errors.append(row["expected_away_goals"] - row["actual_away_goals"])
        total_goal_errors.append(
            row["expected_home_goals"] + row["expected_away_goals"] - row["actual_total_goals"]
        )

    all_goal_errors = home_errors + away_errors
    metrics: dict[str, Any] = {
        "one_x_two": {
            "log_loss": log_loss / len(evaluated),
            "brier_score": brier_score / len(evaluated),
            "accuracy": correct / len(evaluated),
            "calibration": _calibration(evaluated),
        },
        "goal_prediction": {
            "mae": sum(abs(error) for error in all_goal_errors) / len(all_goal_errors),
            "rmse": sqrt(sum(error * error for error in all_goal_errors) / len(all_goal_errors)),
            "home_goals_mae": sum(abs(error) for error in home_errors) / len(home_errors),
            "away_goals_mae": sum(abs(error) for error in away_errors) / len(away_errors),
            "total_goals_mae": sum(abs(error) for error in total_goal_errors) / len(total_goal_errors),
            "total_goals_rmse": sqrt(sum(error * error for error in total_goal_errors) / len(total_goal_errors)),
        },
        "market_brier_scores": {
            key: _brier_binary(
                [row[f"{key}_probability"] for row in evaluated],
                [row[f"actual_{key}"] for row in evaluated],
            )
            for key in ("over_1_5", "over_2_5", "over_3_5", "btts")
        },
    }
    return metrics, evaluated
