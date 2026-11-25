# Phase 3C — Comparator B and Isolated Preprocessing

**Status:** Implemented and verified. No model was trained, no predictions or evaluation metrics were generated, and PostgreSQL was read only.

## Result

Comparator B now has a separate builder which queries `FINISHED` matches only, explicitly excludes awarded fixture 542704 before any team-history calculation, and reuses the original v1 feature builder and six-column allowlist. That retains v1's team-ID history across competitions and venue-specific definitions. It preserves undefined goal averages as null and counts as zero, as in v1.

The generated dataset has **2,699 rows and 25 columns**: nine match-information fields, the exact six original Poisson inputs, and ten targets. It has 2,556 training fixtures and 143 test fixtures. Ordered IDs match the approved v2 dataset exactly and the test IDs match the preserved Poisson v1 test manifest. Fixture 542704 is absent from B targets and historical state.

The six audit-identified Toulouse away fixtures now have one fewer prior away appearance than the legacy dataset. Their cleaned appearances / goals-for average / goals-against average are:

| Fixture | Prior away appearances | Goals for avg | Goals against avg |
|---:|---:|---:|---:|
| 559710 | 16 | 1.3750 | 1.3750 |
| 559684 | 17 | 1.4118 | 1.4118 |
| 559673 | 18 | 1.4444 | 1.4444 |
| 559661 | 19 | 1.4737 | 1.4737 |
| 559645 | 20 | 1.5500 | 1.4000 |
| 559623 | 21 | 1.4762 | 1.4286 |

Each of the four goal-average inputs has 131 missing values (4.85%), where history is unavailable. Both appearance-count inputs have no missing values. Missing goal averages are not replaced with zero.

## Preprocessing and output safeguards

`ml/training/fold_preprocessing.py` defines the four approved chronological folds and a reusable train-only fitting workflow. Numeric values use training medians, training-fitted missing indicators and scaling. Categorical values use a training-fitted missing category and one-hot encoding; unseen validation categories are ignored. `prepare_rolling_fold` splits by date, fits on the training partition only, and transforms validation without refitting. Targets and IDs are rejected as preprocessing inputs. Historical state remains precomputed chronologically in the dataset and is not refitted by this transformer.

Observed eligible row counts for the four folds (training / validation): **954 / 682**, **1,636 / 421**, **2,057 / 313**, and **2,370 / 186**. The second fold has 421 validation rows because awarded fixture 542704 is excluded.

`ml/training/model_outputs.py` provides explicit future B and v2 destinations and rejects the protected existing Poisson, Logistic Regression, and XGBoost v1 artifact directories. No model artifact directory was written in this phase.

## Hashes and reproducibility

Regenerating B twice produced the same dataset and manifest hashes.

| Item | SHA-256 |
|---|---|
| Comparator B CSV | `b33d54eb7da5eed619d97b054d1ca29031f8533bf3bf2b9729f1a4dc7661d09e` |
| Comparator B manifest file | `94cdd5cde9beeabfa255c381f8a2efb3a8f7a66ab4e53c4e94286f3a3eae1eb9` |
| Eligible source records | `dba9ed91da48bf8c802e32cb714733d7c570a8f7736f531ad16cf6a9dc527e4f` |
| Approved v2 source dataset | `7656a4694f315f02913e55260d7599f7034ab27b5a671ca10ffd902375a461d5` |
| Comparator B configuration | `b124b0f78f82f13f582df41cc19c875b9883266615406035dc3d0073a8f7d00d` |
| Ordered fixture IDs | `3d0ed7374de9e70f017fe373e3e313d1a6c49bac8bcfed58ac1436a10ed0d843` |
| Ordered training IDs | `241b7ccf3b5387ac49ab0d3eba9a062e6501a7a53da14e927b7b01e5a5b98bda` |
| Ordered test IDs | `b028450b259b28cf0acbcf13c7200cb62fe7b24032a0006e4e375a33ad618683` |

Hash-based regression tests also confirmed the protected Poisson v1 model/evaluation/predictions and approved v2 CSV/manifest remain at their audited hashes. The database record for 542704 was not changed.

## Verification

- `py -m pytest -q` from `backend` — **64 passed** after all implementation changes.
- `py -m app.jobs.build_comparator_b_dataset` from `backend` — succeeded twice; each run produced 2,699 rows and the same CSV hash.
- Independent CSV/manifest check — no duplicate fixture IDs or duplicate rows; exact ordered ID and split equality with v2; all six legacy inputs only; awarded fixture absent; dataset and source hashes match the manifest.
- No model training, prediction generation, frontend build, database writes, or dependency changes were performed.

## Files created

- `backend/app/jobs/build_comparator_b_dataset.py`
- `backend/tests/test_comparator_b.py`
- `ml/features/comparator_b.py`
- `ml/training/fold_preprocessing.py`
- `ml/training/model_outputs.py`
- `ml/data/processed/comparator_b/v1/training_dataset.csv`
- `ml/data/processed/comparator_b/v1/manifest.json`
- `ml/evaluation/feature_architecture_redesign/phase_3c_implementation_report.md`

## Remaining boundary

This phase supplies clean B data, train-fold-only preprocessing, and protected output locations. It intentionally does not implement or run a B/v2 trainer, rolling-state regeneration per fold, feature ablations, model metrics, or predictions. Model training should wait for review of this implementation and the generated manifest.
