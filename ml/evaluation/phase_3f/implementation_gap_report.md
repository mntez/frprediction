# Phase 3F — Prospective Evaluation Implementation Gaps

**Audit only:** No implementation, model run, prediction generation, database access/write, or frontend change was performed.

| Capability | Current evidence | Gap before prospective evaluation |
|---|---|---|
| Versioned model loading | A, B and C artifacts are file-based under `ml/models/poisson_baseline/`; C includes model metadata, config, preprocessing and two joblib estimators. | No backend registry, artifact checksum allowlist, active-version selection, compatibility check or inference loader. Need immutable model-version records and safe loading by registered version/hash. |
| Reproducible pre-match features | Offline feature generation is implemented in `ml/features/feature_engineering_v2.py` and `backend/app/jobs/build_training_dataset_v2.py`; the job queries completed raw fixtures/results and writes a historical CSV. | No production as-of feature service for an upcoming fixture; no persisted source watermark or exact feature snapshot associated with a forecast. Rebuilding from the mutable current database could rewrite historical context. |
| Prediction API and execution | `backend/app/main.py` creates the FastAPI app and starts the football-data client; no routers or prediction routes are registered. Existing ML code is offline training/evaluation. | No prediction request/service contract, frozen model selection or scheduled pre-kickoff inference job. |
| Prediction persistence | The only ORM tables are competitions, seasons, teams, fixtures and match results; migration `20261122_0001` creates those raw-data tables. | No model-version, prediction, feature-snapshot, prediction-revision, evaluation, or result-verification audit tables. No append-only/unique canonical forecast policy. |
| Prediction-to-result matching | `FixtureRecord.provider_id` is unique and `MatchResultRecord.fixture_id` points to the local fixture, so raw provider results can be associated with fixtures. | No prediction record linked to fixture ID, no result verification/stabilization process, no late/corrected-result history, and no scoring job. |
| Historical prediction audit | Historical evaluation outputs are CSV/JSON artifacts for retrospective model comparisons. | There is no timestamped live forecast ledger. Retrospective backfills must not be presented as forecasts captured before kickoff. Need capture timestamps, actual kickoff, provider fixture snapshot, model/code/config hashes, exact feature JSON and hash, and source availability watermark. |
| Monitoring and comparison | Offline metrics exist for fixed folds and the locked test. | No prospective coverage/late-forecast monitoring, paired-cohort funnel, calibration drift report, per-competition monitoring, uncertainty computation over live predictions, or promotion gate enforcement. |

## Minimum build sequence before prospective collection

1. Define a model registry and immutable A/C artifact manifest with feature-schema and runtime compatibility checks.
2. Implement a point-in-time feature service that enforces the existing same-UTC-date exclusion and records exactly which verified historical data was available at forecast time.
3. Add append-only model-version, prediction and feature-snapshot records via migrations; persist raw probabilities and goal estimates without overwriting prior forecasts.
4. Add pre-kickoff forecast scheduling and a paired A/C capture process with explicit missed/late prediction statuses.
5. Add result verification, fixture-ID association, append-only correction records and delayed scoring.
6. Implement the frozen prospective evaluation report, paired block uncertainty, minimum-sample enforcement and promotion checklist from `prospective_validation_plan.md`.
7. Test replay/audit guarantees: past forecasts are read from stored snapshots, never regenerated; same fixture and model version cannot silently replace a prior forecast; every score links to the exact forecast and verified result revision.

Do not begin collection intended for promotion decisions until the timestamped, immutable forecast ledger and as-of feature path are in place. Do not promote or deploy solely because a retrospective metric or short prospective sample is favorable.

## Files inspected

- `AGENTS.md`, `ml/README.md`
- `ml/baselines/poisson.py`, `ml/features/feature_engineering_v2.py`, `ml/train_poisson_comparison.py`
- `ml/models/poisson_baseline/v1/model.json`
- `ml/models/poisson_baseline/expanded_history/v1/training_config.json`
- `ml/models/poisson_baseline/v2/training_config.json`, `ml/models/poisson_baseline/v2/model_metadata.json`
- `ml/evaluation/poisson_baseline/v1/evaluation.json`
- `ml/evaluation/poisson_baseline/expanded_history/v1/comparison_report.md`, `ml/evaluation/poisson_baseline/expanded_history/v1/final_evaluation.json`
- `ml/evaluation/poisson_baseline/v2/comparison_report.md`, `ml/evaluation/poisson_baseline/v2/final_evaluation.json`
- `ml/evaluation/poisson_baseline/phase_3e/v1/phase_3e_report.md`, `ml/evaluation/poisson_baseline/phase_3e/v1/metric_reconciliation_addendum.md`
- `backend/README.md`, `backend/app/main.py`, `backend/app/models.py`, `backend/app/core/config.py`, `backend/app/db/models.py`, `backend/app/db/session.py`
- `backend/app/services/ingestion.py`, `backend/app/jobs/ingest.py`, `backend/app/jobs/build_training_dataset.py`, `backend/app/jobs/build_training_dataset_v2.py`
- `backend/migrations/env.py`, `backend/migrations/versions/20261122_0001_initial_football_data.py`

Also searched `backend/app` for API router registration, prediction endpoints, model-version references and prediction persistence. The search found no router registration or prediction implementation.
