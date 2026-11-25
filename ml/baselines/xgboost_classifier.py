from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier

from ml.baselines.logistic_regression import FittedPreprocessor, target_class


XGBOOST_PARAMETERS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "n_estimators": 100,
    "max_depth": 2,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "reg_alpha": 0.0,
    "tree_method": "hist",
    "n_jobs": 1,
    "random_state": 0,
    "verbosity": 0,
}


def fit_xgboost(
    training_rows: list[dict[str, Any]],
) -> tuple[FittedPreprocessor, XGBClassifier]:
    if not training_rows:
        raise ValueError("Cannot fit XGBoost without training rows")
    preprocessor = FittedPreprocessor.fit(training_rows)
    features = preprocessor.transform(training_rows)
    labels = [target_class(row) for row in training_rows]
    if set(labels) != {0, 1, 2}:
        raise ValueError("Training split must contain home wins, draws, and away wins")
    classifier = XGBClassifier(**XGBOOST_PARAMETERS)
    classifier.fit(features, labels)
    return preprocessor, classifier
