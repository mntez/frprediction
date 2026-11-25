# Phase 3E — Feature Ablation, Calibration and Robustness

**Scope:** Rolling-origin validation only. The locked 143-match final test predictions were read only to verify artifact integrity/IDs; no final-test scores or outcomes were used for this analysis, tuning or calibration. Existing Phase 3D B/C artifacts remain unchanged.

## Phase 3D baseline verification

Reproduced the saved fold metrics from exact Phase 3D predictions under `ml/evaluation/poisson_baseline/expanded_history/v1` and `ml/evaluation/poisson_baseline/v2`. Validation sample counts were 682, 421, 313, 186. Dataset hashes: B `b33d54eb7da5eed619d97b054d1ca29031f8533bf3bf2b9729f1a4dc7661d09e`, C `7656a4694f315f02913e55260d7599f7034ab27b5a671ca10ffd902375a461d5`. Phase 3D prediction, evaluation, training configuration and model artifact hashes are recorded in `configuration.json` under `source_output_hashes`.

## Controlled estimator comparison

Both feature sets use two `PoissonRegressor` goal models (`alpha=1`, intercept, max_iter=1000, tol=1e-7), fold-only median imputation, missing indicators and scaling; identical outer-fold IDs; and the same Poisson score grid/rate clamp. B uses the original six inputs; C uses the 13 approved inputs. This is a cleaner feature comparison than Phase 3D, while still not proving causal feature effects.

| Fold | N | B LL / Brier / Acc | C LL / Brier / Acc | B total MAE/RMSE | C total MAE/RMSE |
|---|---:|---|---|---:|---:|
| fold_1 | 682 | 1.0601 / 0.6390 / 42.52% | 1.0254 / 0.6143 / 48.24% | 1.2841 / 1.6006 | 1.2796 / 1.5926 |
| fold_2 | 421 | 1.0489 / 0.6312 / 44.89% | 1.0202 / 0.6104 / 49.41% | 1.2813 / 1.6377 | 1.2874 / 1.6337 |
| fold_3 | 313 | 1.0494 / 0.6319 / 44.73% | 1.0068 / 0.6011 / 48.24% | 1.4719 / 1.8273 | 1.4675 / 1.8073 |
| fold_4 | 186 | 1.0191 / 0.6096 / 53.76% | 0.9908 / 0.5894 / 53.23% | 1.5131 / 1.9269 | 1.5479 / 1.9282 |

Complete home/away/total goal metrics, calibration bins and feature lists are saved per fold under `controlled_estimator/`.

## C feature ablation

A finite plan was run: full 13-feature C; leave-one-group-out for five approved groups; and leave-one-feature-out for each of the 13 inputs. No combinatorial search or final-test selection was done.

| Experiment | Features | Mean fold LL | Mean fold Brier | Mean fold Accuracy | Mean fold total-goal MAE |
|---|---:|---:|---:|---:|---:|
| full_13 | 13 | 1.0108 | 0.6038 | 49.78% | 1.3956 |
| drop_group_team_strength | 9 | 1.0172 | 0.6085 | 49.33% | 1.3940 |
| drop_group_form_points | 11 | 1.0099 | 0.6031 | 50.65% | 1.3950 |
| drop_group_recent_goals | 9 | 1.0096 | 0.6030 | 50.37% | 1.3999 |
| drop_group_elo | 12 | 1.0180 | 0.6089 | 49.35% | 1.3950 |
| drop_group_history_exposure | 11 | 1.0119 | 0.6046 | 49.72% | 1.3945 |

All 13 individual leave-one-feature-out experiments (all four folds, exact included features, configuration hashes, metrics and predictions) are saved under `ablations/drop_feature_*/`. The group table is descriptive; the full fold-level records are the auditable results.

## Calibration

Scalar temperature scaling was fitted per outer fold/model using only that fold’s training-set predictions and labels (in-sample), then applied unchanged to validation probabilities. No validation or test labels fit a calibrator. This is allowed training-only fitting but can be optimistic because the calibrator fit predictions are in-sample.

| Model | Fold | T | Raw LL | Calibrated LL | Raw Brier | Calibrated Brier | Raw accuracy | Calibrated accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B | fold_1 | 0.86 | 1.0601 | 1.0613 | 0.6390 | 0.6397 | 42.52% | 42.52% |
| B | fold_2 | 0.89 | 1.0489 | 1.0480 | 0.6312 | 0.6304 | 44.89% | 44.89% |
| B | fold_3 | 0.89 | 1.0494 | 1.0491 | 0.6319 | 0.6317 | 44.73% | 44.73% |
| B | fold_4 | 0.89 | 1.0191 | 1.0150 | 0.6096 | 0.6064 | 53.76% | 53.76% |
| C | fold_1 | 0.84 | 1.0254 | 1.0341 | 0.6143 | 0.6198 | 48.24% | 48.24% |
| C | fold_2 | 0.91 | 1.0202 | 1.0209 | 0.6104 | 0.6107 | 49.41% | 49.41% |
| C | fold_3 | 0.91 | 1.0068 | 1.0046 | 0.6011 | 0.5996 | 48.24% | 48.24% |
| C | fold_4 | 0.92 | 0.9908 | 0.9889 | 0.5894 | 0.5880 | 53.23% | 53.23% |

Reliability bins, counts, forecast probabilities, observed frequencies and classwise ECE are in calibrated/uncalibrated fold metric files. Temperature scaling leaves goal rates and goal errors unchanged.

## Goal prediction diagnosis

The detailed validation-only diagnostics in `goal_analysis.json` include home/away/total expected-goal quantiles and bias; per-competition and actual-outcome scoring means; mean over-market probability vs observed rate; most likely scoreline frequencies; aggregate scoreline and total-goals probability distributions; and clamp/grid sensitivity.

- **B:** predicted mean home/away/total goals 1.583/1.268/2.851; observed 1.620/1.338/2.958; prediction bias home/away/total -0.038/-0.070/-0.107. Rate hits at the configured lower/upper clamp: 0/0.
- **C:** predicted mean home/away/total goals 1.572/1.238/2.810; observed 1.620/1.338/2.958; prediction bias home/away/total -0.048/-0.100/-0.148. Rate hits at the configured lower/upper clamp: 0/0.
- The largest validation 1X2 probability change from using a 0–64 rather than 0–32 score grid was `0`. At λ=8, the single-team probability mass above 32 goals was `3.19e-11`.
- Since the configured rate clamp was not hit in saved validation forecasts, clamping did not alter those forecasts. The grid tail is negligible at observed rates; neither setting explains the higher total-goal error by itself.

The 1X2 probability metrics can improve without improving total-goal MAE/RMSE: Brier/log loss score the ranking of outcome probabilities, while total-goal errors score the sum of the two expected goal rates. Inspect the saved per-competition/outcome residuals and scoreline distributions before attributing the gap to a particular cause.

## Robustness and uncertainty

| Paired comparison | Metric (C minus B / ablation minus full) | Point difference | 95% week-block interval |
|---|---|---:|---:|
| Same-estimator C vs B | log_loss | -0.0083 | [-0.0180, +0.0007] |
| Same-estimator C vs B | brier | -0.0054 | [-0.0120, +0.0007] |
| Same-estimator C vs B | accuracy | -0.0006 | [-0.0167, +0.0169] |
| Same-estimator C vs B | total_goal_abs_error | -0.0160 | [-0.0472, +0.0158] |
| Drop team_strength vs full C | log_loss | +0.0051 | [+0.0024, +0.0074] |
| Drop team_strength vs full C | brier | +0.0038 | [+0.0020, +0.0055] |
| Drop team_strength vs full C | accuracy | -0.0044 | [-0.0123, +0.0033] |
| Drop team_strength vs full C | total_goal_abs_error | +0.0005 | [-0.0045, +0.0057] |
| Drop form_points vs full C | log_loss | -0.0012 | [-0.0021, -0.0004] |
| Drop form_points vs full C | brier | -0.0010 | [-0.0016, -0.0004] |
| Drop form_points vs full C | accuracy | +0.0081 | [+0.0013, +0.0154] |
| Drop form_points vs full C | total_goal_abs_error | -0.0007 | [-0.0015, -0.0000] |
| Drop recent_goals vs full C | log_loss | -0.0010 | [-0.0025, +0.0006] |
| Drop recent_goals vs full C | brier | -0.0006 | [-0.0017, +0.0005] |
| Drop recent_goals vs full C | accuracy | +0.0069 | [+0.0000, +0.0140] |
| Drop recent_goals vs full C | total_goal_abs_error | +0.0036 | [+0.0003, +0.0068] |
| Drop elo vs full C | log_loss | +0.0073 | [+0.0029, +0.0117] |
| Drop elo vs full C | brier | +0.0052 | [+0.0021, +0.0083] |
| Drop elo vs full C | accuracy | -0.0056 | [-0.0145, +0.0033] |
| Drop elo vs full C | total_goal_abs_error | -0.0004 | [-0.0029, +0.0018] |
| Drop history_exposure vs full C | log_loss | +0.0006 | [-0.0000, +0.0013] |
| Drop history_exposure vs full C | brier | +0.0005 | [-0.0000, +0.0009] |
| Drop history_exposure vs full C | accuracy | -0.0006 | [-0.0056, +0.0044] |
| Drop history_exposure vs full C | total_goal_abs_error | -0.0006 | [-0.0052, +0.0040] |

Intervals resample fold-by-ISO-week clusters and preserve pairing, but recurring teams across weeks remain dependent. Four periods and a single season are not strong confirmatory evidence; these intervals are exploratory.

## Hashes, commands and limits

- Phase 3E config SHA-256: `d8ee5716aa08dc3d9b0d04f993e947349b25f283863324f24cad5f9b127a1e1a`
- B dataset SHA-256: `b33d54eb7da5eed619d97b054d1ca29031f8533bf3bf2b9729f1a4dc7661d09e`
- C dataset SHA-256: `7656a4694f315f02913e55260d7599f7034ab27b5a671ca10ffd902375a461d5`
- Ordered validation ID SHA-256: `2a48039a65005130813b527e90c8d913f7ebfa56e1f64ac6868105453b6c2850` (1602 unique validation fixtures)
- Original Phase 3D B/C fold predictions and metrics, final test file hashes, and frozen baseline integrity checks are recorded in the source hash map. No test predictions were scored or analyzed.
- Command: `py -m ml.analysis.phase_3e` from the repository root.
- Tests: `py -m pytest tests/test_phase_3e.py tests/test_poisson_comparison.py tests/test_comparator_b.py -q` (24 passed); full backend suite `py -m pytest -q` (74 passed). No database, frontend, dependencies, approved datasets or existing v1/B/C model artifacts were modified.

## Recommended next experiment

Keep the same locked test untouched. Use a future season or newly accumulated prospective holdout to confirm the controlled same-estimator result, and consider an out-of-fold-trained temperature calibrator before judging calibration. Do not promote a model based on this exploratory phase.
