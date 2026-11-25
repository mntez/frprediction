"""Train-fold-only preprocessing for the preregistered rolling-origin folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.baselines.poisson import TEAM_FEATURES as COMPARATOR_B_FEATURES
from ml.features.feature_engineering_v2 import FEATURE_COLUMNS as V2_FEATURES

V2_CATEGORICAL_FEATURES = ("competition",)
V2_NUMERIC_FEATURES = tuple(V2_FEATURES)
TARGET_OR_METADATA_COLUMNS = frozenset({
    "provider_match_id", "match_date", "home_team", "away_team",
    "home_team_provider_id", "away_team_provider_id", "season_provider_id",
    "season_start_year", "home_goals", "away_goals", "home_win", "draw",
    "away_win", "total_goals", "over_1_5", "over_2_5", "over_3_5", "btts",
})


def comparator_b_preprocessor() -> "FoldPreprocessor":
    """Return the exact original six-input feature transformer for comparator B."""
    return FoldPreprocessor(COMPARATOR_B_FEATURES)


def poisson_v2_preprocessor() -> "FoldPreprocessor":
    """Return the v2 feature transformer with competition context encoded separately."""
    return FoldPreprocessor(V2_NUMERIC_FEATURES, V2_CATEGORICAL_FEATURES)


@dataclass(frozen=True)
class RollingFold:
    name: str
    train_end: str
    validation_start: str
    validation_end: str


ROLLING_ORIGIN_FOLDS = (
    RollingFold("fold_1", "2025-12-31", "2026-01-01", "2026-03-31"),
    RollingFold("fold_2", "2026-03-31", "2026-04-01", "2026-06-30"),
    RollingFold("fold_3", "2026-06-30", "2026-08-01", "2026-09-30"),
    RollingFold("fold_4", "2026-09-30", "2026-10-01", "2026-10-30"),
)


def split_rolling_fold(
    rows: Sequence[dict[str, Any]], fold: RollingFold
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select expanding train and date-bounded validation rows; dates stay intact."""
    train = [row for row in rows if str(row["match_date"]) <= fold.train_end]
    validation = [
        row for row in rows
        if fold.validation_start <= str(row["match_date"]) <= fold.validation_end
    ]
    if not train or not validation:
        raise ValueError(f"{fold.name} has an empty train or validation partition")
    if max(str(row["match_date"]) for row in train) >= min(
        str(row["match_date"]) for row in validation
    ):
        raise ValueError(f"{fold.name} train and validation dates overlap")
    return train, validation


@dataclass(frozen=True)
class PreparedFold:
    training_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]
    training_matrix: np.ndarray
    validation_matrix: np.ndarray
    preprocessor: "FoldPreprocessor"


def prepare_rolling_fold(
    rows: Sequence[dict[str, Any]], fold: RollingFold, preprocessor: "FoldPreprocessor"
) -> PreparedFold:
    """Split chronologically, fit only on train rows, then transform validation."""
    training_rows, validation_rows = split_rolling_fold(rows, fold)
    preprocessor.fit(training_rows)
    return PreparedFold(
        training_rows=training_rows,
        validation_rows=validation_rows,
        training_matrix=preprocessor.transform(training_rows),
        validation_matrix=preprocessor.transform(validation_rows),
        preprocessor=preprocessor,
    )


class FoldPreprocessor:
    """Fit numeric/categorical transformations on training rows, then freeze them."""

    def __init__(self, numeric_features: Sequence[str], categorical_features: Sequence[str] = ()):
        self.numeric_features = tuple(numeric_features)
        self.categorical_features = tuple(categorical_features)
        if not self.numeric_features and not self.categorical_features:
            raise ValueError("At least one feature is required")
        if set(self.numeric_features) & set(self.categorical_features):
            raise ValueError("Numeric and categorical feature lists must be disjoint")
        self.features = (*self.numeric_features, *self.categorical_features)
        forbidden = sorted(set(self.features) & TARGET_OR_METADATA_COLUMNS)
        if forbidden:
            raise ValueError("Targets or identifiers cannot be preprocessing features: " + ", ".join(forbidden))
        self._transformer: ColumnTransformer | None = None

    def _matrix(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        if not rows:
            raise ValueError("Cannot preprocess an empty row set")
        missing = [feature for feature in self.features if feature not in rows[0]]
        if missing:
            raise ValueError("Rows missing preprocessing features: " + ", ".join(missing))
        return np.asarray([[row.get(feature) for feature in self.features] for row in rows], dtype=object)

    def fit(self, training_rows: Sequence[dict[str, Any]]) -> "FoldPreprocessor":
        """Fit using this argument only. Validation rows must be passed to transform."""
        numeric_indices = list(range(len(self.numeric_features)))
        categorical_indices = list(range(len(self.numeric_features), len(self.features)))
        transformers = []
        if numeric_indices:
            numeric = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("numeric", numeric, numeric_indices))
        if categorical_indices:
            categorical = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ])
            transformers.append(("categorical", categorical, categorical_indices))
        self._transformer = ColumnTransformer(transformers, remainder="drop")
        self._transformer.fit(self._matrix(training_rows))
        return self

    def transform(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        if self._transformer is None:
            raise RuntimeError("Fit the preprocessor on training rows before transforming")
        return self._transformer.transform(self._matrix(rows))

    def get_feature_names_out(self) -> np.ndarray:
        if self._transformer is None:
            raise RuntimeError("Preprocessor is not fitted")
        return self._transformer.get_feature_names_out()

    def fitted_state(self) -> dict[str, Any]:
        """Stable, inspectable learned numeric statistics and category vocabulary."""
        if self._transformer is None:
            raise RuntimeError("Preprocessor is not fitted")
        state: dict[str, Any] = {}
        if self.numeric_features:
            pipeline = self._transformer.named_transformers_["numeric"]
            state["numeric_medians"] = pipeline.named_steps["imputer"].statistics_.tolist()
            state["numeric_means"] = pipeline.named_steps["scaler"].mean_.tolist()
            state["numeric_scales"] = pipeline.named_steps["scaler"].scale_.tolist()
        if self.categorical_features:
            pipeline = self._transformer.named_transformers_["categorical"]
            state["categories"] = [values.tolist() for values in pipeline.named_steps["encoder"].categories_]
        return state

