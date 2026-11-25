# Phase 3D — Poisson A/B/C Training and Evaluation

**Date:** 2026-11-24  
**Status:** Completed. A remains frozen; B and C were trained on the approved development data. Final test predictions were generated and evaluated once. These are benchmark models, not production-ready predictions.

## Preflight and methodology

- Preflight passed before fitting: B uses exactly the original six inputs; C uses exactly the approved 13 v2 inputs; metadata and targets are excluded from both feature matrices.
- B and C share the same ordered 2,556 training IDs and 143 test IDs. Their test IDs equal the frozen A IDs. Fixture 542704 is excluded from both datasets, fold partitions and predictions.
- Same-UTC-date groups remain intact. Feature rows only use eligible results from earlier UTC dates; earlier validation outcomes may update historical state for later validation dates, but never model parameters or preprocessing.
- B retains the original multiplicative venue-rate Poisson baseline: train-fold home/away goal means, prior strength 5, 0.05–8.0 rate clamp, and 0–32 goal grid. Missing venue averages back off through the original shrinkage formula.
- C uses two `PoissonRegressor` log-link models (`alpha=1.0`, intercept enabled, `max_iter=1000`, `tol=1e-7`) with the same rate clamp and score grid. Its 13 numeric inputs are median-imputed, missingness-indicated and standardized using training rows only. No feature or hyperparameter search was run.
- B has no external scaling/imputation because that would change the original v1 formula. Its learned global scoring means are fitted from each fold’s training targets only. C receives a new preprocessor instance for every fold and a separate final training-only preprocessor.
- C’s regularized log-link estimator differs from B’s original multiplicative estimator because the approved v2 design specifies independent home/away Poisson regression for the redesigned inputs. Thus B/C controls the fixture population but is not a pure feature-only ablation.

## Final locked test set (n=143)

| Model | Log loss | Brier | Accuracy | Home goals MAE / RMSE | Away goals MAE / RMSE | Total goals MAE / RMSE |
|---|---:|---:|---:|---:|---:|---:|
| A — frozen v1 | 0.9719 | 0.5780 | 51.05% | 0.9577 / 1.2297 | 0.9633 / 1.2907 | 1.4086 / 1.7841 |
| B — expanded history | 1.0052 | 0.6021 | 49.65% | 1.0064 / 1.2327 | 0.9290 / 1.2844 | 1.3891 / 1.7488 |
| C — redesigned v2 | 0.9631 | 0.5715 | 55.24% | 0.9735 / 1.2433 | 0.9383 / 1.2809 | 1.4664 / 1.8225 |

A’s original training fixture IDs were not preserved. A/B therefore compares the frozen 515-row historical baseline with a 2,556-row clean expanded-history model and cannot isolate training-sample size alone. A’s classwise calibration below was recalculated from its saved per-match predictions; its stored v1 metrics and files remain unchanged.

## Rolling-origin validation

| Fold | Validation dates | N | B Log Loss / Brier / Accuracy | C Log Loss / Brier / Accuracy |
|---|---|---:|---|---|
| fold_1 | 2026-01-01–2026-03-22 | 682 | 1.0450 / 0.6273 / 47.95% | 1.0254 / 0.6143 / 48.24% |
| fold_2 | 2026-04-03–2026-05-24 | 421 | 1.0209 / 0.6116 / 49.17% | 1.0202 / 0.6104 / 49.41% |
| fold_3 | 2026-08-07–2026-09-20 | 313 | 1.0077 / 0.6010 / 50.16% | 1.0068 / 0.6011 / 48.24% |
| fold_4 | 2026-10-09–2026-10-30 | 186 | 0.9868 / 0.5857 / 52.15% | 0.9908 / 0.5894 / 53.23% |

### Fold goal MAE / RMSE

| Fold | Model | Home goals | Away goals | Total goals |
|---|---|---:|---:|---:|
| fold_1 | B | 1.0013 / 1.2483 | 0.8814 / 1.1079 | 1.3501 / 1.6863 |
| fold_1 | C | 0.9629 / 1.1917 | 0.8395 / 1.0614 | 1.2796 / 1.5926 |
| fold_2 | B | 0.9744 / 1.1948 | 0.8610 / 1.0853 | 1.2698 / 1.6399 |
| fold_2 | C | 0.9896 / 1.2217 | 0.8312 / 1.0444 | 1.2874 / 1.6337 |
| fold_3 | B | 1.0004 / 1.2506 | 0.9680 / 1.2659 | 1.4108 / 1.7699 |
| fold_3 | C | 1.0334 / 1.2856 | 0.9810 / 1.2729 | 1.4675 / 1.8073 |
| fold_4 | B | 1.0416 / 1.3140 | 1.0336 / 1.3546 | 1.5626 / 1.9984 |
| fold_4 | C | 1.0455 / 1.3314 | 1.0034 / 1.2942 | 1.5479 / 1.9282 |

### Classwise calibration ECE (10 probability bins)

| Period | N | Model | Home win | Draw | Away win |
|---|---:|---|---:|---:|---:|
| Final test | 143 | A | 0.0575 | 0.0415 | 0.0352 |
| Final test | 143 | B | 0.0957 | 0.0700 | 0.0638 |
| Final test | 143 | C | 0.0671 | 0.0400 | 0.0632 |
| fold_1 | 682 | B | 0.0405 | 0.0455 | 0.0332 |
| fold_1 | 682 | C | 0.0440 | 0.0381 | 0.0145 |
| fold_2 | 421 | B | 0.0361 | 0.0346 | 0.0440 |
| fold_2 | 421 | C | 0.0367 | 0.0320 | 0.0357 |
| fold_3 | 313 | B | 0.0632 | 0.0330 | 0.0455 |
| fold_3 | 313 | C | 0.0793 | 0.0163 | 0.0605 |
| fold_4 | 186 | B | 0.0592 | 0.0131 | 0.0583 |
| fold_4 | 186 | C | 0.0368 | 0.0290 | 0.0610 |

Per-class reliability-bin counts, mean forecast probabilities and observed event rates are in each fold’s `metrics.json` and in `final_evaluation.json`.

## Interpretation

- **A vs B:** B is worse on final 1X2 log loss (1.0052 vs 0.9719), Brier (0.6021 vs 0.5780) and accuracy (49.65% vs 51.05%). B has lower total-goals MAE/RMSE, but A’s missing original training IDs prevent a clean attribution to expanded history alone.
- **B vs C:** On the same train/test population, C is better on final 1X2 log loss (0.9631 vs 1.0052), Brier (0.5715 vs 0.6021) and accuracy (55.24% vs 49.65%). C’s total-goals MAE/RMSE are worse (1.4664 / 1.8225 vs 1.3891 / 1.7488). The rolling folds show mixed accuracy and goal results; this test-set result is not proof of general improvement.
- **A vs C:** C is modestly better on final 1X2 log loss, Brier and accuracy, but worse on total-goals MAE/RMSE. The fixed 143-match test period is small, so treat the differences cautiously.

## Integrity and reproducibility

- All B/C fold predictions match their validation fixture manifests with no duplicate, missing or awarded fixture IDs. Final B and C predictions each contain the same 143 ordered IDs as A, and every 1X2 probability triple sums to 1 within tolerance.
- C model and preprocessing artifacts load successfully. The final C fit reported no convergence warnings. The fitted C preprocessing state and per-fold training-ID hashes are saved alongside outputs.
- The original Poisson v1 model, evaluation and prediction hashes and the approved v2 CSV/manifest hashes passed regression checks after training. B’s approved CSV and manifest were not modified.
- Training and test ID hashes, configuration hash, dataset hashes and model artifact hashes are recorded in the two `run_manifest.json` files. No random split or tuning was performed.

| Hash | Value |
|---|---|
| B dataset SHA-256 | `b33d54eb7da5eed619d97b054d1ca29031f8533bf3bf2b9729f1a4dc7661d09e` |
| C dataset SHA-256 | `7656a4694f315f02913e55260d7599f7034ab27b5a671ca10ffd902375a461d5` |
| Configuration SHA-256 | `544baa11f00faefac53aec835a622cbcc1aebfc07d1cc9bdb3e294fff66993f0` |
| Ordered training IDs SHA-256 | `241b7ccf3b5387ac49ab0d3eba9a062e6501a7a53da14e927b7b01e5a5b98bda` |
| Ordered test IDs SHA-256 | `b028450b259b28cf0acbcf13c7200cb62fe7b24032a0006e4e375a33ad618683` |
| Model file hashes | See `run_manifest.json` under each isolated evaluation directory. |

## Commands and tests

- `py -m ml.train_poisson_comparison` (repository root) — preflight passed; trained all four rolling folds and the final B/C models; evaluated each fold and the 143-match final test once.
- `py -m pytest tests/test_poisson_comparison.py tests/test_comparator_b.py tests/test_feature_engineering_v2.py tests/test_poisson_baseline.py -q` (from `backend`) — 41 passed.
- `py -m pytest -q` (from `backend`) — 71 passed.

No PostgreSQL writes, frontend changes, dependency additions or v1 artifact writes were made. Remaining caveats: A’s original training IDs are unavailable; the final test is small; C and B use different Poisson formulations as specified by the approved design; and comparative metrics do not establish production readiness.
