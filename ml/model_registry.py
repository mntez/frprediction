"""Immutable research-model registry and integrity-checked artifact loader.

This module loads registered A/C artifacts only. It does not train, predict,
activate, or fall back between models.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import re
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.baselines.poisson import PoissonBaseline, TEAM_FEATURES
from ml.features.feature_engineering_v2 import FEATURE_COLUMNS as C_FEATURES
from ml.training.fold_preprocessing import FoldPreprocessor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = REPOSITORY_ROOT / "ml" / "models"
A_FEATURES = tuple(TEAM_FEATURES)
C_FEATURES_ORDERED = tuple(C_FEATURES)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


class RegistryError(RuntimeError):
    """Base error for registry lookup and artifact validation failures."""


class UnregisteredModelError(RegistryError):
    pass


class ArtifactIntegrityError(RegistryError):
    pass


class ModelContractError(RegistryError):
    pass


class RuntimeCompatibilityError(RegistryError):
    pass


class ApprovalStatus(StrEnum):
    RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DEPRECATED = "DEPRECATED"


class ActivationStatus(StrEnum):
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ArtifactSpec:
    role: str
    relative_path: str
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("Artifact role is required")
        if not _HASH_RE.fullmatch(self.sha256):
            raise ValueError(f"Invalid SHA-256 for artifact role {self.role}")


@dataclass(frozen=True)
class SourceCodeSpec:
    role: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.role or not _HASH_RE.fullmatch(self.sha256):
            raise ValueError("Source-code role and valid SHA-256 are required")


@dataclass(frozen=True)
class RuntimeRequirement:
    distribution: str
    minimum_inclusive: str | None = None
    maximum_exclusive: str | None = None
    required: bool = True


@dataclass(frozen=True)
class TrainingRuntimeMetadata:
    """Known-at-training values only; None explicitly means not recorded."""

    python: str | None = None
    scikit_learn: str | None = None
    joblib: str | None = None
    numpy: str | None = None
    scipy: str | None = None
    metadata_status: str = "UNKNOWN_NOT_RECORDED"


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_name: str
    semantic_version: str
    source_model_version: str
    model_family: str
    inference_contract: str
    configuration_version: str
    feature_schema_version: str
    feature_names: tuple[str, ...]
    artifacts: tuple[ArtifactSpec, ...]
    source_code: tuple[SourceCodeSpec, ...]
    training_dataset_sha256: str
    training_cutoff: str
    training_runtime: TrainingRuntimeMetadata
    runtime_requirements: tuple[RuntimeRequirement, ...]
    preprocessor_identity: str | None
    preprocessor_sha256: str | None
    approval_status: ApprovalStatus = ApprovalStatus.RESEARCH_CANDIDATE
    activation_status: ActivationStatus = ActivationStatus.INACTIVE
    provenance: Mapping[str, Any] | None = None
    configuration_sha256: str | None = None
    transformed_feature_count: int | None = None

    def __post_init__(self) -> None:
        if not _SEMVER_RE.fullmatch(self.semantic_version):
            raise ValueError("semantic_version must use MAJOR.MINOR.PATCH")
        if not _HASH_RE.fullmatch(self.training_dataset_sha256):
            raise ValueError("training_dataset_sha256 must be a SHA-256 digest")
        if self.preprocessor_sha256 and not _HASH_RE.fullmatch(self.preprocessor_sha256):
            raise ValueError("preprocessor_sha256 must be a SHA-256 digest")
        if self.configuration_sha256 and not _HASH_RE.fullmatch(self.configuration_sha256):
            raise ValueError("configuration_sha256 must be a SHA-256 digest")
        if len(set(self.feature_names)) != len(self.feature_names) or not self.feature_names:
            raise ValueError("feature_names must be a non-empty ordered unique schema")
        if self.provenance is not None:
            object.__setattr__(self, "provenance", _freeze(self.provenance))

    @property
    def key(self) -> tuple[str, str]:
        return self.model_name, self.semantic_version

    @property
    def artifact_hashes(self) -> Mapping[str, str]:
        return MappingProxyType({artifact.role: artifact.sha256 for artifact in self.artifacts})


class ModelRegistry:
    """Trusted, explicit registry; never populated from request payloads."""

    def __init__(self, entries: tuple[ModelRegistryEntry, ...] | list[ModelRegistryEntry]):
        indexed: dict[tuple[str, str], ModelRegistryEntry] = {}
        for entry in entries:
            if entry.key in indexed:
                raise ValueError(f"Duplicate registered model version: {entry.key}")
            indexed[entry.key] = entry
        self._entries = MappingProxyType(indexed)

    def get(self, model_name: str, semantic_version: str) -> ModelRegistryEntry:
        try:
            return self._entries[(model_name, semantic_version)]
        except KeyError as exc:
            raise UnregisteredModelError(
                f"Model version is not registered: {model_name}@{semantic_version}"
            ) from exc

    def entries(self) -> tuple[ModelRegistryEntry, ...]:
        return tuple(self._entries.values())


def _artifact(role: str, path: str, digest: str, media_type: str) -> ArtifactSpec:
    return ArtifactSpec(role, path, digest, media_type)


POISSON_A = ModelRegistryEntry(
    model_name="poisson_a",
    semantic_version="1.0.0",
    source_model_version="v1",
    model_family="multiplicative_poisson",
    inference_contract="poisson_a_multiplicative_v1",
    configuration_version="poisson_a_config_v1",
    feature_schema_version="legacy_venue_features_v1",
    feature_names=A_FEATURES,
    artifacts=(
        _artifact(
            "model_parameters",
            "poisson_baseline/v1/model.json",
            "77b138a6014601b19ee32021da3ce80039cafe82270d8d890b524636d3facd1a",
            "application/json",
        ),
    ),
    source_code=(
        SourceCodeSpec(
            "multiplicative_formula",
            "ml/baselines/poisson.py",
            "076014cfdcd9287338ed5d8a28d8a03cb1e568392cc3db12375a77e2e6133fa7",
        ),
    ),
    training_dataset_sha256="fef1f4a64294baf6fe93c833f13ab15bc0409e05d058c7520073ef2d58a1a718",
    training_cutoff="2026-10-31",
    training_runtime=TrainingRuntimeMetadata(),
    runtime_requirements=(),
    preprocessor_identity=None,
    preprocessor_sha256=None,
    provenance={"training_rows": 515, "training_date_start": "2025-08-15", "training_date_end": "2026-10-30"},
)

_C_FEATURES = (
    "home_attack_strength",
    "home_defense_strength",
    "away_attack_strength",
    "away_defense_strength",
    "home_form_points_5",
    "away_form_points_5",
    "home_goals_for_5",
    "home_goals_against_5",
    "away_goals_for_5",
    "away_goals_against_5",
    "elo_difference",
    "home_history_matches",
    "away_history_matches",
)

POISSON_C = ModelRegistryEntry(
    model_name="poisson_c",
    semantic_version="2.0.0",
    source_model_version="v2",
    model_family="regularized_poisson_log_link",
    inference_contract="poisson_c_two_goal_models_v2",
    configuration_version="poisson_c_config_v2",
    feature_schema_version="football_features_v2",
    feature_names=_C_FEATURES,
    artifacts=(
        _artifact(
            "home_goals_estimator",
            "poisson_baseline/v2/home_goals_model.joblib",
            "435dc6a174dfefbeaf6f31c54c5648beff6fe2598b62acd28a0dd0b9721892d7",
            "application/x-joblib",
        ),
        _artifact(
            "away_goals_estimator",
            "poisson_baseline/v2/away_goals_model.joblib",
            "aada530c957fac5f38372f78bcf59d353d8664e5b122d3a665448032a878c415",
            "application/x-joblib",
        ),
        _artifact(
            "preprocessor",
            "poisson_baseline/v2/preprocessor.joblib",
            "eedf6fd40fe5a7c8dc19978979072a71bf567ff9da7c0609a2c7802b0f69ea99",
            "application/x-joblib",
        ),
        _artifact(
            "model_metadata",
            "poisson_baseline/v2/model_metadata.json",
            "daf56c22f6f6cb1cf6055ced1c27c7402c3f987205717746456f71a05e8c7a1f",
            "application/json",
        ),
        _artifact(
            "training_configuration",
            "poisson_baseline/v2/training_config.json",
            "b2e4087105ad8b077dcf5bcfe74ecd476c3570f6d24afea973c342f3c3dfcb08",
            "application/json",
        ),
        _artifact(
            "training_fixture_manifest",
            "poisson_baseline/v2/fixture_manifest.json",
            "a076edef3f61024262e0554d908d0deec9336831e83c9d8177088413dd95204a",
            "application/json",
        ),
    ),
    source_code=(
        SourceCodeSpec(
            "feature_generation_contract",
            "ml/features/feature_engineering_v2.py",
            "55ac1b291965a31b39fb6a58142e5450c83d3d854a037413aa25c3e87631da7a",
        ),
        SourceCodeSpec(
            "preprocessor_implementation",
            "ml/training/fold_preprocessing.py",
            "3b92a905c9f42bf58485dc941aee92d357798098a20e450dcf0a8b5318e2d397",
        ),
        SourceCodeSpec(
            "poisson_output_contract",
            "ml/train_poisson_comparison.py",
            "e7459f2b2a991e449c4c8c7b3d1424aa2e6da5be1ef1e554d3aa7759c1bf451f",
        ),
    ),
    training_dataset_sha256="7656a4694f315f02913e55260d7599f7034ab27b5a671ca10ffd902375a461d5",
    training_cutoff="2026-10-31",
    training_runtime=TrainingRuntimeMetadata(),
    runtime_requirements=(RuntimeRequirement("scikit-learn", "1.7", "2.0"),),
    preprocessor_identity="ml.training.fold_preprocessing.FoldPreprocessor",
    preprocessor_sha256="eedf6fd40fe5a7c8dc19978979072a71bf567ff9da7c0609a2c7802b0f69ea99",
    provenance={
        "training_rows": 2556,
        "training_date_start": "2025-08-08",
        "training_date_end": "2026-10-30",
        "training_fixture_ids_sha256": "241b7ccf3b5387ac49ab0d3eba9a062e6501a7a53da14e927b7b01e5a5b98bda",
        "historical_runtime_metadata": "UNKNOWN_NOT_RECORDED",
    },
    configuration_sha256="544baa11f00faefac53aec835a622cbcc1aebfc07d1cc9bdb3e294fff66993f0",
    transformed_feature_count=23,
)

DEFAULT_MODEL_REGISTRY = ModelRegistry((POISSON_A, POISSON_C))


@dataclass(frozen=True)
class RuntimeSnapshot:
    python: str
    packages: Mapping[str, str | None]


@dataclass(frozen=True)
class LoadedPoissonA:
    registry_entry: ModelRegistryEntry
    estimator: PoissonBaseline
    feature_names: tuple[str, ...]
    artifact_hashes: Mapping[str, str]
    source_code_hashes: Mapping[str, str]
    runtime: RuntimeSnapshot
    runtime_provenance_warning: str


@dataclass(frozen=True)
class LoadedPoissonC:
    registry_entry: ModelRegistryEntry
    home_goals_estimator: PoissonRegressor
    away_goals_estimator: PoissonRegressor
    preprocessor: FoldPreprocessor
    feature_names: tuple[str, ...]
    transformed_feature_count: int
    artifact_hashes: Mapping[str, str]
    source_code_hashes: Mapping[str, str]
    runtime: RuntimeSnapshot
    runtime_provenance_warning: str


LoadedModel = LoadedPoissonA | LoadedPoissonC


def _safe_relative_path(relative: str) -> PurePosixPath:
    if not relative or "\\" in relative:
        raise ArtifactIntegrityError("Registry paths must be non-empty POSIX-relative paths")
    path = PurePosixPath(relative)
    windows_path = PureWindowsPath(relative)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ArtifactIntegrityError("Absolute artifact paths are not allowed")
    if any(part in ("", ".", "..") for part in path.parts):
        raise ArtifactIntegrityError("Path traversal is not allowed in registry entries")
    return path


def _resolve_under(root: Path, relative: str) -> Path:
    safe = _safe_relative_path(relative)
    try:
        resolved_root = root.resolve(strict=True)
        candidate = (resolved_root / Path(*safe.parts)).resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ArtifactIntegrityError(f"Registered artifact is missing: {relative}") from exc
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ArtifactIntegrityError("Resolved artifact path escapes its configured root") from exc
    if not candidate.is_file():
        raise ArtifactIntegrityError(f"Registered artifact is not a file: {relative}")
    return candidate


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_verified(root: Path, spec: ArtifactSpec | SourceCodeSpec) -> bytes:
    path = _resolve_under(root, spec.relative_path)
    data = path.read_bytes()
    actual = _sha256(data)
    if actual != spec.sha256:
        raise ArtifactIntegrityError(
            f"SHA-256 mismatch for {spec.role}: expected {spec.sha256}, received {actual}"
        )
    return data


def _json_artifact(data: bytes, role: str) -> dict[str, Any]:
    try:
        parsed = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelContractError(f"Invalid JSON in registered {role}") from exc
    if not isinstance(parsed, dict):
        raise ModelContractError(f"Registered {role} must contain a JSON object")
    return parsed


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^\s*(\d+(?:\.\d+)*)", value)
    if not match:
        raise RuntimeCompatibilityError(f"Unrecognized package version: {value}")
    return tuple(int(part) for part in match.group(1).split("."))


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def current_runtime_snapshot() -> RuntimeSnapshot:
    packages = {
        "scikit-learn": _package_version("scikit-learn"),
        "joblib": _package_version("joblib"),
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
    }
    return RuntimeSnapshot(platform.python_version(), MappingProxyType(packages))


def _check_runtime(entry: ModelRegistryEntry, runtime: RuntimeSnapshot) -> None:
    for requirement in entry.runtime_requirements:
        version = runtime.packages.get(requirement.distribution)
        if version is None:
            if requirement.required:
                raise RuntimeCompatibilityError(
                    f"Required runtime package is missing: {requirement.distribution}"
                )
            continue
        parsed = _version_tuple(version)
        if requirement.minimum_inclusive and parsed < _version_tuple(requirement.minimum_inclusive):
            raise RuntimeCompatibilityError(
                f"{requirement.distribution} {version} is below supported minimum "
                f"{requirement.minimum_inclusive}"
            )
        if requirement.maximum_exclusive and parsed >= _version_tuple(requirement.maximum_exclusive):
            raise RuntimeCompatibilityError(
                f"{requirement.distribution} {version} is outside supported range "
                f"(<{requirement.maximum_exclusive})"
            )
    if entry.model_family == "regularized_poisson_log_link":
        for package in ("joblib", "numpy", "scipy"):
            if runtime.packages.get(package) is None:
                raise RuntimeCompatibilityError(f"Required runtime package is missing: {package}")


def _validate_entry(entry: ModelRegistryEntry, access_mode: str) -> None:
    if entry.approval_status in (ApprovalStatus.REJECTED, ApprovalStatus.DEPRECATED):
        raise ModelContractError(f"Model version is not loadable: {entry.approval_status.value}")
    if access_mode == "production" and (
        entry.approval_status != ApprovalStatus.APPROVED
        or entry.activation_status != ActivationStatus.ACTIVE
    ):
        raise ModelContractError("Model version is not approved and active for production")
    if access_mode not in ("research", "production"):
        raise ValueError("access_mode must be 'research' or 'production'")

    roles = {artifact.role for artifact in entry.artifacts}
    code_roles = {source.role for source in entry.source_code}
    if len(roles) != len(entry.artifacts) or len(code_roles) != len(entry.source_code):
        raise ModelContractError("Registry artifact and source-code roles must be unique")
    if entry.model_family == "multiplicative_poisson":
        if entry.inference_contract != "poisson_a_multiplicative_v1":
            raise ModelContractError("Unsupported A inference contract")
        if entry.feature_schema_version != "legacy_venue_features_v1":
            raise ModelContractError("Unsupported A feature schema")
        if entry.feature_names != A_FEATURES or roles != {"model_parameters"}:
            raise ModelContractError("A registry feature/artifact contract mismatch")
        if code_roles != {"multiplicative_formula"}:
            raise ModelContractError("A formula source hash is required")
        if entry.preprocessor_identity is not None or entry.preprocessor_sha256 is not None:
            raise ModelContractError("A does not use an external fitted preprocessor")
    elif entry.model_family == "regularized_poisson_log_link":
        if entry.inference_contract != "poisson_c_two_goal_models_v2":
            raise ModelContractError("Unsupported C inference contract")
        if entry.feature_schema_version != "football_features_v2":
            raise ModelContractError("Unsupported C feature schema")
        required_roles = {
            "home_goals_estimator", "away_goals_estimator", "preprocessor",
            "model_metadata", "training_configuration", "training_fixture_manifest",
        }
        required_code_roles = {
            "feature_generation_contract", "preprocessor_implementation", "poisson_output_contract",
        }
        if entry.feature_names != C_FEATURES_ORDERED or roles != required_roles:
            raise ModelContractError("C registry feature/artifact contract mismatch")
        if code_roles != required_code_roles:
            raise ModelContractError("C source-code hashes are incomplete")
        if not entry.preprocessor_identity or not entry.preprocessor_sha256:
            raise ModelContractError("C requires an explicit fitted preprocessor identity and hash")
        preprocessor_spec = next((spec for spec in entry.artifacts if spec.role == "preprocessor"), None)
        if preprocessor_spec is None or preprocessor_spec.sha256 != entry.preprocessor_sha256:
            raise ModelContractError("C preprocessor hash metadata does not match its artifact")
        if entry.transformed_feature_count != 23:
            raise ModelContractError("C transformed feature count does not match the registered contract")
    else:
        raise ModelContractError(f"Unsupported registered model family: {entry.model_family}")


def _validate_a_parameters(entry: ModelRegistryEntry, model: dict[str, Any]) -> PoissonBaseline:
    if model.get("model_version") != entry.source_model_version:
        raise ModelContractError("A model version metadata mismatch")
    if model.get("model_type") != "independent_team_strength_poisson":
        raise ModelContractError("A model type mismatch")
    if tuple(model.get("team_features_used", ())) != entry.feature_names:
        raise ModelContractError("A ordered feature schema mismatch")
    if model.get("dataset_sha256") != entry.training_dataset_sha256:
        raise ModelContractError("A training dataset hash mismatch")
    if model.get("cutoff_date") != entry.training_cutoff:
        raise ModelContractError("A training cutoff mismatch")
    parameters = model.get("model_parameters")
    means = model.get("training_goal_means")
    method = model.get("methodology")
    if not isinstance(parameters, dict) or not isinstance(means, dict) or not isinstance(method, dict):
        raise ModelContractError("A required model parameters are missing")
    if means.get("home") != parameters.get("home_goal_mean") or means.get("away") != parameters.get("away_goal_mean"):
        raise ModelContractError("A training means are inconsistent")
    if (
        parameters.get("prior_matches") != 5.0
        or parameters.get("minimum_goal_rate") != 0.05
        or parameters.get("maximum_goal_rate") != 8.0
        or parameters.get("max_goals") != 32
        or method.get("prior_matches") != 5.0
        or method.get("goal_rate_clamp") != [0.05, 8.0]
        or method.get("score_probability_grid_max_goals") != 32
    ):
        raise ModelContractError("A inference configuration mismatch")
    try:
        return PoissonBaseline(
            home_goal_mean=float(parameters["home_goal_mean"]),
            away_goal_mean=float(parameters["away_goal_mean"]),
            prior_matches=float(parameters["prior_matches"]),
            max_goals=int(parameters["max_goals"]),
            minimum_goal_rate=float(parameters["minimum_goal_rate"]),
            maximum_goal_rate=float(parameters["maximum_goal_rate"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelContractError("A numerical model parameters are invalid") from exc


def _validate_c_json(
    entry: ModelRegistryEntry,
    metadata: dict[str, Any],
    config: dict[str, Any],
    fixture_manifest: dict[str, Any],
) -> None:
    if metadata.get("model_version") != entry.source_model_version:
        raise ModelContractError("C model version metadata mismatch")
    if metadata.get("model_type") != "two_independent_regularized_poisson_log_link_models":
        raise ModelContractError("C model type mismatch")
    if tuple(metadata.get("features_used", ())) != entry.feature_names:
        raise ModelContractError("C metadata feature order mismatch")
    if tuple(config.get("features", ())) != entry.feature_names:
        raise ModelContractError("C training configuration feature order mismatch")
    if metadata.get("dataset_sha256") != entry.training_dataset_sha256:
        raise ModelContractError("C training dataset hash mismatch")
    if metadata.get("configuration_sha256") != entry.configuration_sha256:
        raise ModelContractError("C canonical configuration hash mismatch")
    config_settings = {
        key: value for key, value in config.items() if key not in {"features", "preprocessing"}
    }
    if metadata.get("settings") != config_settings:
        raise ModelContractError("C model settings and training configuration differ")
    expected_settings = {
        "estimator": "sklearn.linear_model.PoissonRegressor",
        "alpha": 1.0,
        "fit_intercept": True,
        "max_iter": 1000,
        "tol": 1e-7,
        "link": "log",
        "goal_rate_clamp": [0.05, 8.0],
        "score_probability_grid_max_goals": 32,
    }
    for key, expected in expected_settings.items():
        if config.get(key) != expected:
            raise ModelContractError(f"C training configuration mismatch for {key}")
    if fixture_manifest.get("cutoff_date") != entry.training_cutoff:
        raise ModelContractError("C training fixture manifest cutoff mismatch")
    if fixture_manifest.get("training_count") != metadata.get("training_rows"):
        raise ModelContractError("C training row count mismatch")
    if fixture_manifest.get("training_count") != 2556:
        raise ModelContractError("C training population does not match registered provenance")
    expected_ids_hash = (entry.provenance or {}).get("training_fixture_ids_sha256")
    if fixture_manifest.get("training_fixture_ids_sha256") != expected_ids_hash:
        raise ModelContractError("C training fixture ID manifest hash mismatch")
    if 542704 not in fixture_manifest.get("excluded_fixture_ids", ()):
        raise ModelContractError("C awarded fixture exclusion is absent from its training manifest")


def _validate_c_objects(
    entry: ModelRegistryEntry,
    metadata: dict[str, Any],
    home_model: Any,
    away_model: Any,
    preprocessor: Any,
) -> tuple[PoissonRegressor, PoissonRegressor, FoldPreprocessor]:
    for role, model in (("home", home_model), ("away", away_model)):
        if type(model) is not PoissonRegressor:
            raise ModelContractError(f"C {role} estimator has an unexpected type")
        params = model.get_params(deep=False)
        expected = {"alpha": 1.0, "fit_intercept": True, "max_iter": 1000, "tol": 1e-7}
        if any(params.get(key) != value for key, value in expected.items()):
            raise ModelContractError(f"C {role} estimator parameters mismatch")
        if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
            raise ModelContractError(f"C {role} estimator is not fitted")
        if len(model.coef_) != entry.transformed_feature_count:
            raise ModelContractError(f"C {role} estimator input dimension mismatch")

    if type(preprocessor) is not FoldPreprocessor:
        raise ModelContractError("C preprocessing artifact has an unexpected identity")
    if (
        tuple(preprocessor.numeric_features) != entry.feature_names
        or tuple(preprocessor.categorical_features) != ()
        or tuple(preprocessor.features) != entry.feature_names
    ):
        raise ModelContractError("C preprocessor input feature order mismatch")
    transformer = getattr(preprocessor, "_transformer", None)
    if type(transformer) is not ColumnTransformer or not hasattr(transformer, "transformers_"):
        raise ModelContractError("C preprocessor is missing its fitted ColumnTransformer")
    numeric = transformer.named_transformers_.get("numeric")
    if type(numeric) is not Pipeline or tuple(numeric.named_steps) != ("imputer", "scaler"):
        raise ModelContractError("C numeric preprocessing pipeline mismatch")
    imputer = numeric.named_steps["imputer"]
    scaler = numeric.named_steps["scaler"]
    if (
        type(imputer) is not SimpleImputer
        or imputer.strategy != "median"
        or not imputer.add_indicator
        or not imputer.keep_empty_features
        or type(scaler) is not StandardScaler
        or not scaler.with_mean
        or not scaler.with_std
    ):
        raise ModelContractError("C fitted imputer/scaler contract mismatch")
    try:
        output_count = len(preprocessor.get_feature_names_out())
    except Exception as exc:
        raise ModelContractError("C preprocessor has no valid fitted output schema") from exc
    if output_count != entry.transformed_feature_count:
        raise ModelContractError("C transformed feature count mismatch")
    preprocessing_metadata = metadata.get("preprocessing")
    if not isinstance(preprocessing_metadata, dict):
        raise ModelContractError("C preprocessing metadata is missing")
    if len(preprocessing_metadata.get("feature_names_out", ())) != output_count:
        raise ModelContractError("C preprocessing metadata output schema mismatch")
    if preprocessor.fitted_state() != preprocessing_metadata.get("state"):
        raise ModelContractError("C fitted preprocessing artifact and metadata state differ")
    return home_model, away_model, preprocessor


class ModelArtifactLoader:
    """Loads only trusted entries from an injected/static registry."""

    def __init__(
        self,
        registry: ModelRegistry = DEFAULT_MODEL_REGISTRY,
        *,
        model_root: Path = DEFAULT_MODEL_ROOT,
        code_root: Path = REPOSITORY_ROOT,
        runtime_snapshot: RuntimeSnapshot | None = None,
    ) -> None:
        self.registry = registry
        self.model_root = Path(model_root)
        self.code_root = Path(code_root)
        self.runtime_snapshot = runtime_snapshot or current_runtime_snapshot()

    def load(
        self,
        model_name: str,
        semantic_version: str,
        *,
        access_mode: str = "research",
    ) -> LoadedModel:
        entry = self.registry.get(model_name, semantic_version)
        _validate_entry(entry, access_mode)
        _check_runtime(entry, self.runtime_snapshot)

        # Read and verify every registered artifact and source file before any
        # joblib deserialization. Bytes are then deserialized from the verified
        # buffers, avoiding a second path read and a hash/load race.
        artifact_bytes = {spec.role: _read_verified(self.model_root, spec) for spec in entry.artifacts}
        code_bytes = {spec.role: _read_verified(self.code_root, spec) for spec in entry.source_code}
        artifact_hashes = MappingProxyType({role: _sha256(data) for role, data in artifact_bytes.items()})
        code_hashes = MappingProxyType({role: _sha256(data) for role, data in code_bytes.items()})

        if entry.model_family == "multiplicative_poisson":
            model_json = _json_artifact(artifact_bytes["model_parameters"], "A model parameters")
            estimator = _validate_a_parameters(entry, model_json)
            return LoadedPoissonA(
                registry_entry=entry,
                estimator=estimator,
                feature_names=entry.feature_names,
                artifact_hashes=artifact_hashes,
                source_code_hashes=code_hashes,
                runtime=self.runtime_snapshot,
                runtime_provenance_warning="Training runtime versions were not recorded for Poisson A.",
            )

        metadata = _json_artifact(artifact_bytes["model_metadata"], "C model metadata")
        config = _json_artifact(artifact_bytes["training_configuration"], "C training configuration")
        fixture_manifest = _json_artifact(artifact_bytes["training_fixture_manifest"], "C training fixture manifest")
        _validate_c_json(entry, metadata, config, fixture_manifest)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                home_model = joblib.load(io.BytesIO(artifact_bytes["home_goals_estimator"]))
                away_model = joblib.load(io.BytesIO(artifact_bytes["away_goals_estimator"]))
                preprocessor = joblib.load(io.BytesIO(artifact_bytes["preprocessor"]))
        except Exception as exc:
            raise ArtifactIntegrityError("Verified C joblib artifact could not be deserialized") from exc
        if any(issubclass(item.category, InconsistentVersionWarning) for item in caught):
            raise RuntimeCompatibilityError("C artifact was serialized by an incompatible scikit-learn version")
        home_model, away_model, preprocessor = _validate_c_objects(
            entry, metadata, home_model, away_model, preprocessor
        )
        return LoadedPoissonC(
            registry_entry=entry,
            home_goals_estimator=home_model,
            away_goals_estimator=away_model,
            preprocessor=preprocessor,
            feature_names=entry.feature_names,
            transformed_feature_count=entry.transformed_feature_count or 0,
            artifact_hashes=artifact_hashes,
            source_code_hashes=code_hashes,
            runtime=self.runtime_snapshot,
            runtime_provenance_warning=(
                "Training Python/scikit-learn/joblib/NumPy/SciPy versions were not recorded; "
                "the current runtime is within the registered scikit-learn range, but exact "
                "historical serialization compatibility is unproven. Research loading only."
            ),
        )

