"""Explicit isolated output locations for future comparator B and Poisson v2 runs."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_V1_PATHS = (
    REPOSITORY_ROOT / "ml" / "models" / "poisson_baseline" / "v1",
    REPOSITORY_ROOT / "ml" / "models" / "logistic_regression" / "v1",
    REPOSITORY_ROOT / "ml" / "models" / "xgboost" / "v1",
    REPOSITORY_ROOT / "ml" / "evaluation" / "poisson_baseline" / "v1",
    REPOSITORY_ROOT / "ml" / "evaluation" / "logistic_regression" / "v1",
    REPOSITORY_ROOT / "ml" / "evaluation" / "xgboost" / "v1",
)

OUTPUTS = {
    "comparator_b": {
        "model": REPOSITORY_ROOT / "ml" / "models" / "poisson_baseline" / "expanded_history" / "v1",
        "evaluation": REPOSITORY_ROOT / "ml" / "evaluation" / "poisson_baseline" / "expanded_history" / "v1",
    },
    "poisson_v2": {
        "model": REPOSITORY_ROOT / "ml" / "models" / "poisson_baseline" / "v2",
        "evaluation": REPOSITORY_ROOT / "ml" / "evaluation" / "poisson_baseline" / "v2",
    },
}


def assert_unprotected_output_path(path: Path) -> Path:
    resolved = path.resolve()
    for protected in PROTECTED_V1_PATHS:
        root = protected.resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError(f"Refusing to write into protected v1 artifact path: {resolved}")
    return resolved


def get_output_directory(comparison: str, artifact_kind: str) -> Path:
    try:
        path = OUTPUTS[comparison][artifact_kind]
    except KeyError as exc:
        raise ValueError(f"Unknown future output destination: {comparison}/{artifact_kind}") from exc
    return assert_unprotected_output_path(path)


for _group in OUTPUTS.values():
    for _path in _group.values():
        assert_unprotected_output_path(_path)

