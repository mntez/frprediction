# Phase 3F — Model Selection and Prospective Validation Plan

**Status:** Protocol proposal only. No model is promoted or deployed. A, B, C, their datasets, and their historical evaluation artifacts remain frozen.

## 1. Candidate audit

| Candidate | Inputs and estimator | Training population | Existing evaluation | Strengths | Limitations |
|---|---|---|---|---|---|
| **A — Poisson v1 (frozen benchmark)** | Six venue inputs: home home-matches played, home home-goals scored average, home home-goals conceded average, away away-matches played, away away-goals scored average, away away-goals conceded average. Original multiplicative team-strength Poisson formula; training-period overall home/away goal means; shrinkage prior strength 5; rate clamp 0.05–8; score grid 0–32. | 515 rows, 2025-08-15–2026-10-30. Frozen 143-match test. Original training fixture IDs were not preserved. | Final test: 1X2 log loss **0.9719**, Brier **0.5780**, accuracy **51.05%**; home/away/total goals MAE **0.9577/0.9633/1.4086**, RMSE **1.2297/1.2907/1.7841**. | Simple, interpretable, frozen reference; strongest stated historic 1X2 benchmark before Phase 3D. | Smaller training set; six venue-only inputs; missing original training IDs prevent a clean historical population comparison; one small held-out period is not prospective evidence. |
| **B — Expanded-history Poisson** | Same six inputs and original multiplicative Poisson estimator/formula as A. | 2,556 rows, 2025-08-08–2026-10-30; same 143 test fixtures as A and C. | Final test: log loss **1.0052**, Brier **0.6021**, accuracy **49.65%**; home/away/total MAE **1.0064/0.9290/1.3891**, RMSE **1.2327/1.2844/1.7488**. | Isolates a larger eligible training population using A’s feature definitions and estimator more closely than C. | A’s training IDs are unavailable, so A-vs-B cannot attribute differences to sample size alone. B did not beat A on the reported 1X2 metrics. Not a production candidate by itself. |
| **C — Redesigned Poisson v2** | Thirteen inputs: home/away attack and defence strength; home/away five-match points; home/away five-match goals for and against; Elo difference; home/away history exposure. Two independent scikit-learn `PoissonRegressor` log-link estimators, `alpha=1`, intercept, `max_iter=1000`, `tol=1e-7`; train-fitted median imputation, missing indicators and scaling; clamp 0.05–8; score grid 0–32. | Same 2,556 training rows and same 143 test fixtures as B. | Final test: log loss **0.9631**, Brier **0.5715**, accuracy **55.24%**; home/away/total MAE **0.9735/0.9383/1.4664**, RMSE **1.2433/1.2809/1.8225**. Phase 3E.1 found controlled-estimator pooled rolling-validation total-goals MAE B **1.3466**, C **1.3495** (C−B **+0.0029**); the separate Phase 3D B/C goal bootstrap was a different estimator comparison and its 95% interval crossed zero. | Best reported 1X2 scores on the locked historical test; redesigned point-in-time strengths, form and Elo; B/C populations and test IDs are aligned. | Worse final-test goal errors than A/B; limited history and one test period; estimator differs from B; rolling comparisons are exploratory and fold outcomes are mixed. No prospective forecasts or operational lifecycle have been audited in production. |

The 143-match historical test results above are existing, frozen benchmark results, quoted for candidate inventory only. They must not be used to tune this prospective protocol. Keep the Phase 3E.1 reconciliation explicit: its `+0.0029` is for the controlled same-estimator pooled MAE comparison; `−0.0160` is for the different Phase 3D B/C predictions.

## 2. Prospective capture and evaluation protocol

### Before the first forecast

Register a signed/immutable evaluation manifest containing the protocol version, evaluation start and stopping dates, eligible competitions, match eligibility rules, scheduled forecast lead time, A/C model versions and artifact SHA-256 values, feature schema and code revision hashes, prediction/evaluation metric definitions, missing-prediction policy, minimum samples, bootstrap design, and promotion criteria below. B may be retained as a diagnostic comparator but A-vs-C is the promotion comparison. Do not change this registration after outcomes begin accruing.

### Capture one canonical forecast per model and fixture

For each eligible fixture, create A and C forecasts from the same predeclared lead-time window, recommended **60 minutes before scheduled kickoff**. Persist each forecast append-only with:

- provider fixture ID and stable local fixture ID; competition, season, teams, scheduled UTC kickoff, and fixture snapshot/version;
- model name/version, immutable artifact hash, feature schema version, inference-code revision, and inference configuration hash;
- `prediction_created_at_utc`, intended lead time, scheduled and eventual actual kickoff time, and an explicit assertion that prediction time precedes actual kickoff;
- exact ordered feature names and exact unrounded feature values/nulls used, serialized snapshot and SHA-256, feature-generation version, and the source-data/result watermark used to construct that snapshot;
- all published probabilities, expected goals, model outputs, and validation status.

Use the already approved same-UTC-date exclusion: results from any match on a fixture’s UTC date cannot enter that day’s features. More generally, only results whose official status/score had been verified and ingested before the forecast timestamp **and** whose UTC date precedes the target fixture date may affect features. Never rebuild an old forecast from today’s database state. A correction to a saved prediction is a new append-only revision linked to the prior row; the original forecast remains visible and is used for the primary score unless a pre-registered data-integrity rule invalidates it.

If a kickoff moves, retain the original forecast and timestamps. It remains eligible only if it was genuinely generated before the actual kickoff and satisfies the registered lead-time window; otherwise exclude it from the primary paired cohort and record the exclusion. Do not create a retrospective forecast for a missed fixture.

### Freeze and result verification

Freeze A and C model artifacts, preprocessing, feature logic, and inference code for the full comparison period. A model change starts a new prospective evaluation version; do not mix versions. Match the result to the stored provider fixture ID, not team names or approximate dates. Score a prediction only after the provider has a final eligible status and both full-time scores have passed the registered result-validation checks. Keep result source, retrieval/verification timestamps, and any correction history. Do not overwrite forecast snapshots or past evaluation rows after a provider result correction; append a result revision and retain the original scoring audit trail.

### Paired cohort and missing forecasts

The primary comparison includes only fixtures for which both A and C have valid canonical pre-kickoff forecasts and a verified final result. Use exactly the same fixture IDs and actual labels for each model. Report the full fixture funnel: eligible fixtures, both-model forecasts captured in time, result-verified fixtures, exclusions by reason, and final paired sample. A fixture missing either forecast is excluded from both models’ paired metrics, but missing-forecast and late-forecast rates are separately monitored so a model cannot appear better through selective coverage.

## 3. Metrics and uncertainty

For each model and paired cohort report sample count, date range, competition/season counts, coverage, and:

- 1X2 multiclass log loss;
- 1X2 multiclass Brier score, using the same sum/normalization convention for both models;
- 1X2 accuracy;
- classwise home-win, draw and away-win reliability tables/curves with bin counts, mean predicted probability, observed frequency, and classwise ECE;
- home goals, away goals and total goals MAE and RMSE.

Report paired C-minus-A differences with confidence intervals, not only separate model intervals. Use a predeclared cluster/block bootstrap over competition-season and UTC matchweek/date blocks (4,000 replicates with a fixed seed; resample paired fixture rows together within each selected block). Include the number of blocks and matches per competition. Fixtures are not independent: teams recur, and results share dates and competitions, so intervals are exploratory uncertainty estimates rather than guarantees. Also report per-competition and per-season results to expose concentration of any gain.

Calibration bins must be fixed before unblinding or use a documented equal-count binning rule applied identically to each model. Do not fit calibrators during this comparison. If calibration is later proposed, register it as a separate model version trained on pre-evaluation/out-of-fold data before the prospective period starts.

## 4. Duration, sample size and promotion rules

Set the stopping rule before forecasts begin. Require **at least two complete competition seasons**, **at least 2,000 paired verified fixtures overall**, and **at least 250 paired fixtures in each of the six competitions**. The period must also span at least 18 calendar months; continue to a predeclared maximum of 30 months if the per-competition minimum has not been reached. Do not review model-comparison outcomes before the stopping threshold; operational data-quality checks may run without showing comparative performance.

Promote C for a controlled review only if all registered criteria pass:

1. Primary 1X2 log loss improves by at least **2% relative to A**, and the paired 95% block-bootstrap confidence interval for C−A log loss lies wholly below zero.
2. Multiclass Brier point estimate is no worse than A, with the upper 95% interval for C−A no greater than **+0.01**.
3. No home, away or total-goals MAE worsens by more than **0.03 goals**, and none of their RMSE worsens by more than **0.05 goals**, versus A.
4. Classwise calibration is reported for every outcome. No class may have ECE above **0.08** or be worse than A by more than **0.02**. Inspect reliability-bin counts and calibration slope/intercept before interpreting ECE, especially for sparse bins.
5. Criteria 1–4 hold on the full paired period and are not driven by a single competition: report all six competition-level results and flag any with log loss more than 5% worse than A.
6. Data completeness, pre-kickoff timing, model-version consistency, probability validity and immutable snapshot audit checks pass.

Accuracy is reported but is not a standalone promotion criterion. If a criterion fails or the interval remains inconclusive, keep A as the incumbent and do not tune against the same prospective outcomes. Any next experiment must be specified against a new future period before its outcomes are reviewed. Passing these criteria is a recommendation for a separate approval/release decision, not automatic deployment.

## 5. Operational audit requirements before prospective launch

Before collecting launch-quality forecasts, implement and test versioned artifact loading, inference-time v1/v2 feature computation from as-of source state, append-only prediction persistence, stable fixture/result association, result verification and correction history, immutable model registry entries, scheduled forecast generation and monitoring. No current backend prediction API or prediction persistence lifecycle was found; current offline training artifacts and historical CSVs are not sufficient substitutes for this audit trail.

## 6. Boundaries

This is a prospective evaluation plan, not a claim that either model is production-ready. The existing locked test remains untouched for future selection. Historical stored forecasts cannot be relabeled as prospective forecasts.
