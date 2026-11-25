# Phase 3G.2 — Model Registry and Artifact Loader

**Status:** Implemented as an isolated research-only component. No training, estimator prediction, database access/write, migration, model activation, frontend change or dependency addition occurred.

## Verified candidate contracts

| Field | Poisson A | Poisson C |
|---|---|---|
| Registry identity | `poisson_a@1.0.0` (`source_model_version=v1`) | `poisson_c@2.0.0` (`source_model_version=v2`) |
| Family / contract | `multiplicative_poisson` / `poisson_a_multiplicative_v1` | `regularized_poisson_log_link` / `poisson_c_two_goal_models_v2` |
| Schema/config versions | `legacy_venue_features_v1` / `poisson_a_config_v1` | `football_features_v2` / `poisson_c_config_v2` |
| Ordered inputs | `home_home_matches_played`, `home_home_goals_scored_avg`, `home_home_goals_conceded_avg`, `away_away_matches_played`, `away_away_goals_scored_avg`, `away_away_goals_conceded_avg` | `home_attack_strength`, `home_defense_strength`, `away_attack_strength`, `away_defense_strength`, `home_form_points_5`, `away_form_points_5`, `home_goals_for_5`, `home_goals_against_5`, `away_goals_for_5`, `away_goals_against_5`, `elo_difference`, `home_history_matches`, `away_history_matches` |
| Dataset hash / cutoff | `fef1f4a64294baf6fe93c833f13ab15bc0409e05d058c7520073ef2d58a1a718` / `2026-10-31` | `7656a4694f315f02913e55260d7599f7034ab27b5a671ca10ffd902375a461d5` / `2026-10-31` |
| Preprocessor | None; the original A multiplicative formula uses its stored means/prior and six raw inputs. | `ml.training.fold_preprocessing.FoldPreprocessor`; fitted 13-input transform to 23 columns. Not refit by this loader. |
| Candidate state | `RESEARCH_CANDIDATE`, `INACTIVE` | `RESEARCH_CANDIDATE`, `INACTIVE` |

A's JSON contains the model parameters and configuration, including home/away means, shrinkage prior 5, rate clamp `[0.05, 8.0]`, and score grid 0–32. The loader validates these values and the formula-source hash before constructing the existing `PoissonBaseline` parameters object. It does not substitute a regression model or add preprocessing.

C's two fitted `PoissonRegressor` objects are checked for type, fitted coefficients and configured alpha/intercept/iteration/tolerance values. The exact saved preprocessor is checked for ordered numeric features, median imputation with missing indicators, standard scaling, fitted state parity with metadata, and 23 output columns. No `.fit()` or `.predict()` call is made.

## Registered artifacts and hashes

Hashes below are SHA-256 of the exact files. All paths are relative to the configured model root `ml/models/`.

| Model | Relative path | Type | SHA-256 |
|---|---|---|---|
| A | `poisson_baseline/v1/model.json` | JSON parameters/config | `77b138a6014601b19ee32021da3ce80039cafe82270d8d890b524636d3facd1a` |
| C | `poisson_baseline/v2/home_goals_model.joblib` | fitted scikit-learn `PoissonRegressor` | `435dc6a174dfefbeaf6f31c54c5648beff6fe2598b62acd28a0dd0b9721892d7` |
| C | `poisson_baseline/v2/away_goals_model.joblib` | fitted scikit-learn `PoissonRegressor` | `aada530c957fac5f38372f78bcf59d353d8664e5b122d3a665448032a878c415` |
| C | `poisson_baseline/v2/preprocessor.joblib` | fitted `FoldPreprocessor` | `eedf6fd40fe5a7c8dc19978979072a71bf567ff9da7c0609a2c7802b0f69ea99` |
| C | `poisson_baseline/v2/model_metadata.json` | JSON estimator/schema metadata | `daf56c22f6f6cb1cf6055ced1c27c7402c3f987205717746456f71a05e8c7a1f` |
| C | `poisson_baseline/v2/training_config.json` | JSON training config | `b2e4087105ad8b077dcf5bcfe74ecd476c3570f6d24afea973c342f3c3dfcb08` |
| C | `poisson_baseline/v2/fixture_manifest.json` | JSON training ID/provenance manifest | `a076edef3f61024262e0554d908d0deec9336831e83c9d8177088413dd95204a` |

The registry also pins source implementation hashes: A formula `ml/baselines/poisson.py` (`076014cfdcd9287338ed5d8a28d8a03cb1e568392cc3db12375a77e2e6133fa7`); C feature generation `ml/features/feature_engineering_v2.py` (`55ac1b291965a31b39fb6a58142e5450c83d3d854a037413aa25c3e87631da7a`); C preprocessing implementation `ml/training/fold_preprocessing.py` (`3b92a905c9f42bf58485dc941aee92d357798098a20e450dcf0a8b5318e2d397`); and C's saved comparison/output-contract implementation `ml/train_poisson_comparison.py` (`e7459f2b2a991e449c4c8c7b3d1424aa2e6da5be1ef1e554d3aa7759c1bf451f`). Code paths are rooted at the repository root and independently hash-verified.

## Runtime metadata and compatibility

The installed environment used for read-only loader verification reports:

- Python `3.14.6`
- scikit-learn `1.9.1`
- joblib `1.6.0`
- NumPy `2.5.5`
- SciPy `1.18.1`

The project requirement `scikit-learn>=1.7,<2.0` is the only recorded supported version range applied by the registry. C also requires joblib, NumPy and SciPy to be present. The loader rejects scikit-learn outside the recorded range and rejects sklearn's `InconsistentVersionWarning` during deserialization.

**Training-time versions are unknown.** Neither A nor C records the historical Python, scikit-learn, joblib, NumPy or SciPy versions. The registry explicitly stores those values as `UNKNOWN_NOT_RECORDED`; it does not infer them from the current environment. C was loadable and its preprocessor contract validated under the current runtime, but exact training-runtime serialization compatibility is not established. Both candidates remain research-only; this loader does not certify production readiness.

## Implementation

`ml/model_registry.py` contains frozen typed entries, artifact/source-code specs, schema/config versions, dataset hashes/cutoffs, runtime provenance, and separate approval/activation metadata. The static registry includes only A and C; B is not registered. The loader:

1. Looks up the exact `(model_name, semantic_version)` and raises `UnregisteredModelError` if absent. It never falls back.
2. Validates research/production access mode, model-family-specific artifact roles, schemas and contract metadata. Production access rejects these unapproved/inactive candidates.
3. Rejects absolute paths, traversal and resolved symlinks outside the configured root.
4. Reads and verifies every required artifact and code file against registered SHA-256 before deserializing anything. C joblib files are deserialized from the same verified in-memory bytes, avoiding a separate read/hash race.
5. Validates A's JSON/formula contract or C's estimator, fitted-preprocessor, feature-order, training-manifest and configuration contracts.
6. Captures current runtime versions while preserving missing historical runtime metadata as unknown.

The registry is static trusted application configuration, not a client-writeable API or database registry. Hashes provide integrity relative to that trusted registry; signing/managed approval and deployment events are future work. No public API, DB model, migration, dynamic activation, scheduler or prediction lifecycle was added.

## Tests and commands

From `backend`:

- `py -m pytest tests/test_model_registry.py -q` — **14 passed, 1 skipped**.
- `py -m pytest -q` — **88 passed, 1 skipped**.

Coverage includes valid A/C loading from isolated temporary artifact copies; unregistered versions and no fallback; missing/modified files; traversal; unsupported schemas; runtime incompatibility; missing preprocessor; model-contract mismatch; candidate rejection in production mode; and protected artifact immutability. The symlink escape test was skipped because creating a symlink is not permitted in this Windows test environment; the path traversal test passed, and the loader separately resolves and confines symlinks where the OS permits them.

## Files changed or created

- Created `ml/model_registry.py`.
- Created `backend/tests/test_model_registry.py`.
- Created `ml/evaluation/phase_3g/phase_3g2_implementation_report.md`.

No existing model, evaluation, dataset, feature-generation, backend application, database, frontend or dependency files were changed.

## Protected artifact fingerprints

Fingerprints combine sorted relative paths and each file's SHA-256. The following values were recorded before implementation and verified again after tests and report generation:

| Protected tree | Files | SHA-256 fingerprint |
|---|---:|---|
| A model | 1 | `7aef45397fc004889741fedf13797e8d7e48fddf95916faa088f1a3363dbf2a3` |
| A evaluation | 2 | `a36ce4eab8fd2ea4fbe36990472c457f59a7dd377aa02717f3e3b7ad1afccc85` |
| B model | 3 | `09a636dddfccf221d2e8b930aedb66df6bb01dde453c0b90525154a5abba158f` |
| B evaluation | 23 | `8357889dd104f874b933968677d8c95e450c6980ad8daf5ce870bfc25ba69109` |
| C model | 6 | `d292ee74372db39e3d987a9d2e6db8e0797076b57279296386c4fe070908b83f` |
| C evaluation | 24 | `008133b7a8b6cb455e42eccc6aa86288a6f2b35ed607ddc2b77ec7bdcefc5dec` |
| B dataset | 2 | `c5b2f9f6eb7076fa07949c29e5f23828f207e966fa666b0f55e66249d203739c` |
| C dataset | 2 | `20b8a443023b6a30b650641d68f205ba7f188b2b6c7dfa4b7a3b8f356e732451` |
| `ml/evaluation/poisson_baseline/phase_3e/v1` | 215 | `3d82f74c080af630dee89afd640f315d9c27093f2c7d10e64d382a23ffdd6bde` |
| `ml/models/poisson_baseline/phase_3e/v1` | 80 | `3db795800dc4ae637fe31dc478191feb6796d4c076c562c44ed6f05961dfdbfa` |

The Phase 3E evaluation fingerprint was transcribed incorrectly in the initial Phase 3G.2 report. The pre-existing Phase 3G.1.1 `architecture_clarification_report.md` contains the authoritative 215-file value shown above, and a fresh calculation matches it. The older Phase 3E.1 addendum records a separate 214-file snapshot made before that addendum was added; it is not the baseline for the current 215-file tree. Thus equality to the authoritative 215-file baseline is established. The repository has no `.git` directory, so Git history is unavailable; no backup files were found. Other listed protected-tree fingerprints also match their recorded baselines.
