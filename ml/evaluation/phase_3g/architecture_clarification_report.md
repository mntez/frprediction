# Phase 3G.1.1 — Architecture Clarification Audit

**Scope:** Read-only architecture and artifact audit, with documentation updates only. No model training, prediction generation, database access/write, migrations, code, frontend, dependency, dataset or protected artifact changes were performed.

## 1. Result availability timestamps

### Confirmed existing capabilities

- `FixtureRecord.retrieved_at` and `MatchResultRecord.retrieved_at` exist, but ingestion overwrites them on update. They are last local retrieval/upsert times, not first-seen or verification history.
- Provider `Match` includes scheduled `utcDate`, `lastUpdated`, status and score fields. The current schema has no explicit actual completion timestamp.
- Ingestion updates fixture kickoff/status and current full-time score, but does not preserve provider `lastUpdated` in the current result record or append source-observation history.
- Existing historical rows cannot establish when the system first learned each final score. No historic availability time can be reconstructed safely.

### Proposed changes

1. Append a `provider_result_observations` row for each successfully fetched result-bearing provider observation with `fetched_at` (server UTC receipt time), payload hash, status/scores, provider update timestamp if supplied, provider fixture ID, and fetch run ID.
2. Define `first_seen_at` as the earliest locally recorded `fetched_at` where the observation has an eligible ordinary final status and valid full-time scores. This is first local observation only. Historical backfill starts observation from rollout/import time; never backdate it.
3. Define `verified_at` as server UTC time when the named validation procedure confirms the observation and fixture-ID mapping. A `verified_result_revisions` row points to that observation and stores validation rule/version and verifier identity.
4. Keep `match_completed_at` nullable. Populate it only when an authoritative source explicitly provides an actual completion timestamp and preserve the source/provenance. `utcDate`, `lastUpdated`, status transition observation and retrieval time are not substitutes.
5. Permit a result in prediction-time historical state only when the exact result revision was fetched and verified at or before `prediction_created_at_utc`, has eligible `FINISHED` status, and its kickoff UTC date is strictly before the target fixture date. Save the revision IDs and source watermark in the feature snapshot.

This creates reliable system-knowledge timestamps going forward. It does not create historical availability facts. Forecasts before the observation ledger exists cannot claim prospective availability and cannot be reconstructed from current database state.

## 2. Model artifact audit

The C artifacts exist at the paths below and their hashes match the Phase 3D run manifest where the manifest covers those artifacts. Estimator objects were loaded for type/parameter inspection; the saved preprocessor transformed one schema-shaped in-memory row to a 23-column matrix. No estimator prediction or model regeneration was performed.

| Candidate component | Path | Confirmed type/format | SHA-256 |
|---|---|---|---|
| A model parameters | `ml/models/poisson_baseline/v1/model.json` | JSON | `77b138a6014601b19ee32021da3ce80039cafe82270d8d890b524636d3facd1a` |
| C home model | `ml/models/poisson_baseline/v2/home_goals_model.joblib` | `sklearn.linear_model.PoissonRegressor`; alpha 1, intercept, max_iter 1000, tol 1e-7 | `435dc6a174dfefbeaf6f31c54c5648beff6fe2598b62acd28a0dd0b9721892d7` |
| C away model | `ml/models/poisson_baseline/v2/away_goals_model.joblib` | `sklearn.linear_model.PoissonRegressor`; alpha 1, intercept, max_iter 1000, tol 1e-7 | `aada530c957fac5f38372f78bcf59d353d8664e5b122d3a665448032a878c415` |
| C fitted preprocessor | `ml/models/poisson_baseline/v2/preprocessor.joblib` | `ml.training.fold_preprocessing.FoldPreprocessor`; 13 raw numeric features to 23 transformed columns | `eedf6fd40fe5a7c8dc19978979072a71bf567ff9da7c0609a2c7802b0f69ea99` |
| C model metadata | `ml/models/poisson_baseline/v2/model_metadata.json` | JSON | `daf56c22f6f6cb1cf6055ced1c27c7402c3f987205717746456f71a05e8c7a1f` |
| C training config | `ml/models/poisson_baseline/v2/training_config.json` | JSON | `b2e4087105ad8b077dcf5bcfe74ecd476c3570f6d24afea973c342f3c3dfcb08` |
| C preprocessing audit state | `ml/evaluation/poisson_baseline/v2/final_preprocessing.json` | JSON | `51bb07b1f887d121dec13feaca4a8a30217ba7e7feb1cd043df8b23444244db2` |
| C run manifest | `ml/evaluation/poisson_baseline/v2/run_manifest.json` | JSON; binds home/away/preprocessor hashes and training configuration | `1fce94e4e8bcfac5eac21e38722c5d43deabc4d91391ccc7035d20866ee05157` |

**C inference contract:** the ordered 13 inputs are in `model_metadata.json.features_used`. `preprocessor.joblib` is the fitted transform; it median-imputes expected missing numeric values with missing indicators, scales, and returns 23 columns. The separately fitted home and away Poisson regressors return rates. The compatible output adapter applies `[0.05,8.0]` rate clamping and the 0–32 independent-Poisson score grid for outcome/market probabilities and most likely score. The registry must verify the three joblib hashes, JSON metadata/config hashes, exact feature order and estimator settings before loading.

**A inference contract:** `model.json` stores the two training goal means, prior strength 5, `[0.05,8.0]` clamp, 0–32 grid, model version/date information and six original inputs. `ml.baselines.poisson.PoissonBaseline` implements the original multiplicative formula after venue-rate shrinkage. A has no fitted preprocessing artifact: inference requires the A JSON parameters, exactly the six named raw venue features, and the matching formula implementation. Never run A through C's scaler or regression models.

**Missing reproducibility evidence:**

- The current environment uses scikit-learn **1.9.1** and successfully loaded the estimator/preprocessor objects. `backend/requirements.txt` allows `scikit-learn>=1.7,<2.0`; that range is not an exact serialization/runtime guarantee.
- The saved run/model metadata does not record training Python, scikit-learn, joblib, NumPy or SciPy versions. Exact training-runtime compatibility is therefore unconfirmed, although current-environment artifact loading was confirmed.
- Register source-code hashes as part of the model bundle. Current inspected hashes: `ml/baselines/poisson.py` `076014cfdcd9287338ed5d8a28d8a03cb1e568392cc3db12375a77e2e6133fa7`; `ml/features/feature_engineering_v2.py` `55ac1b291965a31b39fb6a58142e5450c83d3d854a037413aa25c3e87631da7a`; `ml/training/fold_preprocessing.py` `3b92a905c9f42bf58485dc941aee92d357798098a20e450dcf0a8b5318e2d397`; `ml/train_poisson_comparison.py` `e7459f2b2a991e449c4c8c7b3d1424aa2e6da5be1ef1e554d3aa7759c1bf451f`.
- A's model JSON has a training dataset hash and row/date counts, but its original training fixture ID list is unavailable. This limits exact training population reconstruction, not the stored numerical inference formula.

## 3. Rescheduled fixture identity and scoring

### Proposed changes

- Add append-only `fixture_schedule_revisions`, keyed by fixture and monotonic revision, with kickoff UTC, `fetched_at`, source payload hash and provider update timestamp where available. The first revision is the first schedule observed after tracking begins; do not fabricate older schedule history.
- Add a `forecast_cycle` UUID bound to fixture, evaluation cohort, schedule-revision ID and forecast slot. Unique `(fixture_id, cohort_id, schedule_revision_id, forecast_slot)` makes retries for the same schedule/cycle idempotent.
- Link predictions to a forecast cycle and enforce unique `(forecast_cycle_id, model_version_id)` for one canonical forecast per model per cycle. Different model versions remain pairable; a changed schedule creates a distinct cycle.
- Preserve all prior kickoff snapshots and prediction rows. A newly observed reschedule appends a schedule revision and marks the prior unplayed cycle `SUPERSEDED_BY_RESCHEDULE`; it does not update or delete its forecast.
- For primary scoring, select only the canonical forecast from the latest tracked schedule revision associated with the played fixture that was generated in the registered T−60 window and before actual kickoff. If that revision has no valid paired A/C forecasts, exclude the fixture from paired scoring and report the superseded/missed records in coverage. Never substitute the old schedule's forecast.

## 4. Model-specific registry contract

Use a common typed adapter interface: validate registry/artifacts; build model-specific raw features from the as-of fixture state; transform with that model's saved contract; return a common `PredictionOutput` containing 1X2 probabilities, expected home/away goals, score/market probabilities, model version/artifact/config hashes, and feature snapshot hash.

- **A adapter:** manifest requires A JSON, six-feature schema, exact formula-code hash, prior/means/clamp/grid and output contract. No preprocessing artifact is expected.
- **C adapter:** manifest requires both estimator joblibs, fitted preprocessor joblib, C metadata/config and output adapter/code hashes. It validates ordered 13-feature raw schema, 23-column transformed schema, estimator class/parameters, preprocessor parameters, output clamp/grid and all component hashes.
- Both are registered by immutable version and status; a research candidate is not loadable for public production responses. Model family selects the adapter explicitly. Unsupported runtime/schema, missing artifact or checksum mismatch fails closed without fallback.

## 5. Exact forecast timing policy

**Proposed canonical schedule window:** if `S` is kickoff UTC from the active tracked schedule revision, run at one-minute polling cadence in the inclusive interval `[S−65 minutes, S−55 minutes]` (target T−60). Use timezone-aware server UTC timestamps.

- Earlier than `S−65`: defer; no forecast row.
- Within the interval: create one canonical forecast per registered version, then return the existing row on retry.
- Later than `S−55` but before kickoff: emit `MISSED_WINDOW`; no late substitute in that cycle/cohort.
- At/after scheduled kickoff or after `IN_PLAY` has been observed: emit `TOO_LATE`; never backfill a prospective forecast.
- Kickoff changed before the window: supersede pending cycle and schedule against the new revision.
- Kickoff changed after capture: preserve and supersede old forecast for the primary played fixture; open a new revision/cycle and predict only if its new window is still met. Otherwise record a miss and exclude both models for paired scoring.
- Another forecast for the same fixture/cohort is allowed only for a new schedule revision or a separately preregistered slot/cohort. No second canonical prediction for the same model/cycle/slot.

**Unresolved timing limitation:** football-data.org currently provides scheduled `utcDate`, `lastUpdated` and status, but the existing schema has no verified actual kickoff timestamp. `IN_PLAY` first-observed time proves play had started by then, not exactly when it started. Until a trusted actual-start timestamp is available, the system can prove it forecast within the schedule-based window and before any observed `IN_PLAY` state, but cannot prove actual-start ordering when a match started early or provider status arrived late. Strict cohort handling must either exclude ambiguous cases or be approved by the protocol owner before enrollment.

## Confirmed capabilities, proposed changes and unresolved approvals

| Classification | Finding |
|---|---|
| **Confirmed** | Current result/fixture retrieval timestamps are mutable last-upsert timestamps. Provider schema has scheduled kickoff and `lastUpdated`, but no actual completion field. No fetch/verification history exists. |
| **Confirmed** | A JSON parameters and formula source exist. C has two fitted PoissonRegressor joblibs, its fitted preprocessor joblib, metadata/config and manifest; checksums match. Artifacts load in current Python environment; C's preprocessor transforms 13 raw inputs to 23 columns. |
| **Confirmed** | Current backend has no prediction registry, forecast-cycle/schedule-history records, result verification ledger, or prediction API/persistence. Existing kickoff can be updated by ingestion. |
| **Proposed** | Append-only result-observation timestamps, verified result revisions, schedule revisions, forecast cycles, prediction linkage and model-specific registry adapters as specified above. |
| **Unresolved — approval needed** | Decide whether strict prospective scoring excludes fixtures lacking an authoritative actual kickoff timestamp, or approve another auditable conservative rule. |
| **Unresolved — approval needed** | Approve the `[S−65,S−55]` canonical T−60 window, one forecast per model per cycle, no late replacement, and latest-schedule-revision-only scoring. |
| **Unresolved — implementation prerequisite** | Decide and record the exact Python/scikit-learn/joblib/NumPy/SciPy runtime for C artifact loading; the training environment versions were not captured. Add and freeze source-code hashes before any registered deployment. |
| **Unresolved — approval needed** | Decide how to treat older results that have current final scores but no post-rollout observation timestamps. They cannot be assigned invented first-seen/verified times. |

## Files inspected

- `AGENTS.md`
- `ml/evaluation/phase_3g/prediction_infrastructure_design.md`
- `ml/evaluation/phase_3f/prospective_validation_plan.md`, `ml/evaluation/phase_3f/implementation_gap_report.md`
- `backend/app/models.py`, `backend/app/db/models.py`, `backend/app/services/ingestion.py`, `backend/app/core/config.py`, `backend/requirements.txt`
- `ml/models/poisson_baseline/v1/model.json`
- `ml/models/poisson_baseline/v2/home_goals_model.joblib`, `away_goals_model.joblib`, `preprocessor.joblib`, `model_metadata.json`, `training_config.json`
- `ml/evaluation/poisson_baseline/v2/run_manifest.json`, `final_preprocessing.json`
- `ml/baselines/poisson.py`, `ml/features/feature_engineering_v2.py`, `ml/training/fold_preprocessing.py`, `ml/train_poisson_comparison.py`

## Protected artifact fingerprints

Fingerprints cover sorted relative paths and SHA-256 values for every file in each directory. The Phase 3E evaluation fingerprint includes the preserved reconciliation addendum. These values were recorded before documentation edits and are rechecked after them.

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
| Phase 3E evaluation | 215 | `3d82f74c080af630dee89afd640f315d9c27093f2c7d10e64d382a23ffdd6bde` |
| Phase 3E model diagnostics | 80 | `3db795800dc4ae637fe31dc478191feb6796d4c076c562c44ed6f05961dfdbfa` |

All recorded fingerprints matched after the documentation updates. No tests were run because no executable code changed.
