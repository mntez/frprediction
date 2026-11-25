import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from os import symlink

import pytest

from ml.model_registry import (
    DEFAULT_MODEL_REGISTRY,
    REPOSITORY_ROOT,
    ActivationStatus,
    ArtifactIntegrityError,
    ArtifactSpec,
    ModelArtifactLoader,
    ModelContractError,
    ModelRegistry,
    POISSON_A,
    POISSON_C,
    RuntimeCompatibilityError,
    RuntimeSnapshot,
    UnregisteredModelError,
)


MODEL_ROOT = REPOSITORY_ROOT / "ml" / "models"


def _copy_registered_artifacts(entry, destination: Path) -> Path:
    root = destination / "models"
    root.mkdir(parents=True)
    for artifact in entry.artifacts:
        source = MODEL_ROOT / artifact.relative_path
        target = root / artifact.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return root


def _registry_with(entry):
    return ModelRegistry([entry])


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(path: Path) -> tuple[int, str]:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    payload = "".join(
        f"{item.relative_to(path).as_posix()} {_file_hash(item)}\n" for item in files
    )
    return len(files), hashlib.sha256(payload.encode()).hexdigest()


def test_registered_a_loads_original_multiplicative_contract_from_temp_artifacts(tmp_path):
    root = _copy_registered_artifacts(POISSON_A, tmp_path)
    loaded = ModelArtifactLoader(_registry_with(POISSON_A), model_root=root).load(
        "poisson_a", "1.0.0"
    )

    assert loaded.registry_entry.inference_contract == "poisson_a_multiplicative_v1"
    assert loaded.registry_entry.approval_status.value == "RESEARCH_CANDIDATE"
    assert loaded.registry_entry.activation_status == ActivationStatus.INACTIVE
    assert loaded.feature_names == POISSON_A.feature_names
    assert loaded.estimator.prior_matches == 5.0
    assert loaded.estimator.max_goals == 32
    assert loaded.artifact_hashes["model_parameters"] == POISSON_A.artifacts[0].sha256


def test_registered_c_loads_both_fitted_models_and_preprocessor_without_fitting_or_predicting(tmp_path):
    root = _copy_registered_artifacts(POISSON_C, tmp_path)
    loaded = ModelArtifactLoader(_registry_with(POISSON_C), model_root=root).load(
        "poisson_c", "2.0.0"
    )

    assert loaded.registry_entry.inference_contract == "poisson_c_two_goal_models_v2"
    assert loaded.feature_names == POISSON_C.feature_names
    assert loaded.transformed_feature_count == 23
    assert loaded.preprocessor.features == POISSON_C.feature_names
    assert loaded.home_goals_estimator.get_params()["alpha"] == 1.0
    assert loaded.away_goals_estimator.get_params()["max_iter"] == 1000
    assert loaded.artifact_hashes["preprocessor"] == POISSON_C.preprocessor_sha256
    assert loaded.runtime_provenance_warning.startswith("Training Python/")


def test_unknown_model_version_fails_without_falling_back_to_another_model():
    loader = ModelArtifactLoader()
    with pytest.raises(UnregisteredModelError, match="not registered"):
        loader.load("poisson_c", "9.9.9")
    with pytest.raises(UnregisteredModelError, match="not registered"):
        loader.load("poisson_b", "1.0.0")
    assert {entry.model_name for entry in DEFAULT_MODEL_REGISTRY.entries()} == {
        "poisson_a", "poisson_c"
    }


def test_missing_artifact_fails_closed(tmp_path):
    root = _copy_registered_artifacts(POISSON_A, tmp_path)
    (root / POISSON_A.artifacts[0].relative_path).unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        ModelArtifactLoader(_registry_with(POISSON_A), model_root=root).load(
            "poisson_a", "1.0.0"
        )


def test_modified_hash_fails_before_deserialization(tmp_path):
    root = _copy_registered_artifacts(POISSON_C, tmp_path)
    spec = next(a for a in POISSON_C.artifacts if a.role == "home_goals_estimator")
    path = root / spec.relative_path
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        ModelArtifactLoader(_registry_with(POISSON_C), model_root=root).load(
            "poisson_c", "2.0.0"
        )


def test_path_traversal_is_rejected(tmp_path):
    unsafe = replace(
        POISSON_A,
        artifacts=(replace(POISSON_A.artifacts[0], relative_path="../outside/model.json"),),
    )
    with pytest.raises(ArtifactIntegrityError, match="traversal"):
        ModelArtifactLoader(_registry_with(unsafe), model_root=tmp_path).load(
            "poisson_a", "1.0.0"
        )


def test_symlink_outside_model_root_is_rejected(tmp_path):
    root = _copy_registered_artifacts(POISSON_A, tmp_path)
    artifact_path = root / POISSON_A.artifacts[0].relative_path
    external = tmp_path / "outside.json"
    shutil.copyfile(artifact_path, external)
    artifact_path.unlink()
    try:
        symlink(external, artifact_path)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable in this Windows environment: {exc}")
    with pytest.raises(ArtifactIntegrityError, match="escapes"):
        ModelArtifactLoader(_registry_with(POISSON_A), model_root=root).load(
            "poisson_a", "1.0.0"
        )


def test_symlinked_directory_outside_model_root_is_rejected(tmp_path):
    root = _copy_registered_artifacts(POISSON_A, tmp_path)
    registered_path = Path(POISSON_A.artifacts[0].relative_path)
    internal_directory = root / registered_path.parts[0]
    external_directory = tmp_path / "external-model-directory" / registered_path.parts[0]
    external_directory.parent.mkdir(parents=True)
    shutil.copytree(internal_directory, external_directory)
    shutil.rmtree(internal_directory)
    try:
        symlink(external_directory, internal_directory, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable in this Windows environment: {exc}")

    with pytest.raises(ArtifactIntegrityError, match="escapes"):
        ModelArtifactLoader(_registry_with(POISSON_A), model_root=root).load(
            "poisson_a", "1.0.0"
        )


def test_unsupported_feature_schema_is_rejected(tmp_path):
    root = _copy_registered_artifacts(POISSON_C, tmp_path)
    unsupported = replace(POISSON_C, feature_schema_version="future-v99")
    with pytest.raises(ModelContractError, match="schema"):
        ModelArtifactLoader(_registry_with(unsupported), model_root=root).load(
            "poisson_c", "2.0.0"
        )


def test_runtime_incompatibility_fails_before_deserialization(tmp_path):
    root = _copy_registered_artifacts(POISSON_C, tmp_path)
    runtime = RuntimeSnapshot(
        python="3.14.0",
        packages=MappingProxyType({
            "scikit-learn": "2.0.0",
            "joblib": "1.5.0",
            "numpy": "2.2.0",
            "scipy": "1.15.0",
        }),
    )
    with pytest.raises(RuntimeCompatibilityError, match="outside supported range"):
        ModelArtifactLoader(
            _registry_with(POISSON_C), model_root=root, runtime_snapshot=runtime
        ).load("poisson_c", "2.0.0")


def test_missing_preprocessor_fails_closed(tmp_path):
    root = _copy_registered_artifacts(POISSON_C, tmp_path)
    preprocessor = next(a for a in POISSON_C.artifacts if a.role == "preprocessor")
    (root / preprocessor.relative_path).unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        ModelArtifactLoader(_registry_with(POISSON_C), model_root=root).load(
            "poisson_c", "2.0.0"
        )


def test_c_feature_configuration_mismatch_fails_after_integrity_check(tmp_path):
    root = _copy_registered_artifacts(POISSON_C, tmp_path)
    config_spec = next(a for a in POISSON_C.artifacts if a.role == "training_configuration")
    config_path = root / config_spec.relative_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["features"][0] = "wrong_feature"
    changed = json.dumps(config, separators=(",", ":")).encode()
    config_path.write_bytes(changed)
    artifacts = tuple(
        replace(spec, sha256=hashlib.sha256(changed).hexdigest())
        if spec.role == "training_configuration" else spec
        for spec in POISSON_C.artifacts
    )
    modified_entry = replace(POISSON_C, artifacts=artifacts)
    with pytest.raises(ModelContractError, match="feature order"):
        ModelArtifactLoader(_registry_with(modified_entry), model_root=root).load(
            "poisson_c", "2.0.0"
        )


def test_a_inference_contract_mismatch_is_rejected(tmp_path):
    root = _copy_registered_artifacts(POISSON_A, tmp_path)
    wrong = replace(POISSON_A, inference_contract="poisson_c_two_goal_models_v2")
    with pytest.raises(ModelContractError, match="A inference contract"):
        ModelArtifactLoader(_registry_with(wrong), model_root=root).load(
            "poisson_a", "1.0.0"
        )


def test_missing_registered_artifact_role_is_rejected(tmp_path):
    root = _copy_registered_artifacts(POISSON_C, tmp_path)
    incomplete = replace(
        POISSON_C,
        artifacts=tuple(a for a in POISSON_C.artifacts if a.role != "preprocessor"),
    )
    with pytest.raises(ModelContractError, match="artifact contract"):
        ModelArtifactLoader(_registry_with(incomplete), model_root=root).load(
            "poisson_c", "2.0.0"
        )


def test_production_mode_rejects_research_candidates(tmp_path):
    root = _copy_registered_artifacts(POISSON_A, tmp_path)
    with pytest.raises(ModelContractError, match="not approved and active"):
        ModelArtifactLoader(_registry_with(POISSON_A), model_root=root).load(
            "poisson_a", "1.0.0", access_mode="production"
        )


def test_loading_does_not_modify_existing_a_b_c_or_phase3e_artifacts():
    protected = (
        REPOSITORY_ROOT / "ml/models/poisson_baseline/v1",
        REPOSITORY_ROOT / "ml/models/poisson_baseline/expanded_history/v1",
        REPOSITORY_ROOT / "ml/models/poisson_baseline/v2",
        REPOSITORY_ROOT / "ml/evaluation/poisson_baseline/v1",
        REPOSITORY_ROOT / "ml/evaluation/poisson_baseline/expanded_history/v1",
        REPOSITORY_ROOT / "ml/evaluation/poisson_baseline/v2",
        REPOSITORY_ROOT / "ml/data/processed/comparator_b/v1",
        REPOSITORY_ROOT / "ml/data/processed/v2",
        REPOSITORY_ROOT / "ml/evaluation/poisson_baseline/phase_3e/v1",
        REPOSITORY_ROOT / "ml/models/poisson_baseline/phase_3e/v1",
    )
    before = [_tree_hash(path) for path in protected]
    loader = ModelArtifactLoader()
    loader.load("poisson_a", "1.0.0")
    loader.load("poisson_c", "2.0.0")
    after = [_tree_hash(path) for path in protected]
    assert after == before
