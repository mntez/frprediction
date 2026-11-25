# Historical feature dataset

This stage builds a reproducible supervised dataset from completed match results
already stored in PostgreSQL. It does not train a model and it does not modify
the raw database records.

Run from `backend`:

```powershell
py -m app.jobs.build_training_dataset
```

The generated CSV is written to `ml/data/processed/training_dataset.csv`. Use
`--output PATH` to choose another destination. The output is regenerated
atomically from the current database state.

## Inputs and leakage controls

- Only fixtures marked `FINISHED` or `AWARDED` with a stored match result are
  included. Results attached to cancelled or unfinished statuses are excluded,
  even if the provider response contains score fields.
- Matches are ordered by UTC kickoff time and provider match ID.
- Features for a UTC date are calculated before results from that date enter
  history. This deliberately excludes same-day fixtures, including earlier
  kickoffs, because this schema does not record when a result became available.
- Previous team and head-to-head history is built only from results on earlier
  UTC dates. No standings, post-match statistics, or provider retrieval times
  are used.
- Team history spans the competitions in the database, so prior domestic or
  other competition matches for the same provider team ID can contribute.
- Rolling averages and form use the last five available prior matches.
- Head-to-head values are blank until at least three earlier meetings exist.

## Features and missing values

The output includes competition, season, date, team provider IDs/names;
cumulative prior match/win/draw/loss and goals-for/goals-against counts; rolling
five-match goal averages and recent form; home-only history for the home team;
away-only history for the away team; and optional head-to-head counts.

Counts with no history are `0`. Averages, win rates, recent form, and
head-to-head values with insufficient history are empty CSV cells (missing),
not fabricated zeroes. Targets contain actual home and away goals plus
`home_win`, `draw`, `away_win`, `total_goals`, `over_1_5`, `over_2_5`,
`over_3_5`, and `btts`.

The schema stores kickoff but not final-whistle/completion time. Excluding all
same-UTC-date results is a conservative safeguard; unusual suspended matches
whose completion timing cannot be reconstructed remain a source-data limit.

## Poisson baseline (v1)

The first baseline is a transparent independent-goals Poisson model. It fits
training-period mean home and away goals, then estimates each side's expected
goals from its venue-specific historical scoring average and its opponent's
venue-specific conceding average. Team rates are shrunk toward the corresponding
training-period league mean using a five-match prior. Rates are bounded to
0.05–8.0 goals. Independent Poisson score probabilities are evaluated over a
0–32 goal grid and normalized; these produce 1X2, goal-market, BTTS, and most
likely-score outputs.

The generated source contains 658 eligible completed results and 49 columns
(39 feature/match-information columns and 10 target columns). It excludes one
cancelled match and two in-play rows that had score records in PostgreSQL. The
baseline uses six venue-specific goal-history fields: home team home goals
scored/conceded averages and match count; away team away goals
scored/conceded averages and match count. Global home/away means are estimated
from training targets only. Rolling form and head-to-head are not used by this
baseline.

The split is fixed at **2026-10-31**: training uses dates before the cutoff;
testing uses dates on or after it. This retains whole dates and yields 515
training rows (2025-08-15–2026-10-30) and 143 test rows
(2026-10-31–2026-11-21). A fixed-year split would leave only 16 training rows,
so the date cutoff provides a more useful initial benchmark without shuffling.
Test feature rows use prior results from earlier test-period dates only after
those dates, simulating walk-forward availability; model rates remain fit on
the training period.

Run `py -m ml.train_poisson_baseline` from the repository root. The versioned
estimator is saved to `ml/models/poisson_baseline/v1/model.json`; aggregate
metrics and per-match probability-vs-result predictions are saved under
`ml/evaluation/poisson_baseline/v1/`. Artifacts include a SHA-256 of the source
CSV and the cutoff date.

### v1 chronological test results

| Metric | Result |
|---|---:|
| 1X2 Log Loss | 0.9719 |
| 1X2 Brier Score | 0.5780 |
| 1X2 Accuracy | 51.05% |
| Goal MAE (pooled home/away goals) | 0.9605 |
| Goal RMSE (pooled home/away goals) | 1.2606 |
| Total-goals MAE | 1.4086 |
| Total-goals RMSE | 1.7841 |

Additional probability checks found a top-class expected calibration error of
0.0695. Market Brier scores are saved in the evaluation JSON. Test probabilities
were finite and within 0–1, and each 1X2 distribution summed to 1 within
numeric tolerance.

This is a benchmark, not a production model. The sample is small and limited by
provider data availability; the independent Poisson assumption ignores score
correlation, red cards, lineup changes, and evolving team strength. Later
baselines must use this chronological protocol and comparable metrics before
any model is considered better.

## Logistic Regression baseline (v1)

This is a multinomial scikit-learn Logistic Regression classifier for the
three actual-result classes: home win, draw, and away win. It is trained from
the same processed dataset and exact same cutoff/test match IDs as Poisson v1.
Poisson predictions are used only for the evaluation comparison; they are never
training labels.

The model uses the existing pre-match numeric team-form, goals, venue-specific,
head-to-head, and season-year fields. Competition, home-team name, away-team
name, and recent-form strings are treated as categorical values. Provider IDs,
match IDs, kickoff date, scores, and all target columns are excluded from model
inputs. The `lbfgs` solver, L2 regularization (`C=1`), and a fixed random state
are used for reproducibility.

Preprocessing is fitted only on the 515 training rows. Numeric missing values
are filled with each column's training median and receive a missing-value
indicator; scaled values use the training mean and population standard
deviation. Categorical values are one-hot encoded from training categories, with
a dedicated missing category. A category first seen in the test period is
represented by all-zero columns. The stored preprocessing JSON contains only
training-fitted medians, scaling statistics, and categories.

The split remains cutoff **2026-10-31**, training 2025-08-15–2026-10-30
(515 rows), test 2026-10-31–2026-11-21 (143 rows). Every logistic test match ID
is checked against the saved Poisson test predictions before evaluation.

The model is saved under `ml/models/logistic_regression/v1/` (`model.joblib`,
`model_metadata.json`, and `preprocessing.json`). Evaluation JSON, per-match
probabilities, and the direct comparison report are saved under
`ml/evaluation/logistic_regression/v1/`.

### v1 chronological test results and Poisson comparison

| Model | Log Loss | Brier | Accuracy |
|---|---:|---:|---:|
| Poisson v1 | 0.9719 | 0.5780 | 51.05% |
| Logistic Regression v1 | 1.1635 | 0.6869 | 41.26% |

On this holdout, Logistic Regression is worse on all three summary metrics.
Its per-class precision / recall / F1 are:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Home win | 0.514 | 0.569 | 0.540 | 65 |
| Draw | 0.107 | 0.107 | 0.107 | 28 |
| Away win | 0.442 | 0.380 | 0.409 | 50 |

Confusion matrix (actual rows, predicted columns; order home win, draw, away win):

```text
[[37, 13, 15],
 [16,  3,  9],
 [19, 12, 19]]
```

Top-class expected calibration error is 0.1667 (mean confidence 0.5793 versus
accuracy 0.4126). Full calibration bins and per-match actual/predicted values
are in the saved evaluation artifacts. The model is currently a weaker
benchmark than Poisson on this limited sample, especially for draws; it is not
production-ready. Do not promote it based on this evaluation.

## XGBoost baseline (v1)

XGBoost v1 is a conservative multiclass `XGBClassifier` using the same
pre-match input columns, target classes, processed CSV, 2026-10-31 cutoff, and
143 test match IDs as both earlier baselines. It never uses the other models'
predictions as labels. It checks ordered test IDs against both saved prediction
files before training/evaluation and fails if either baseline's test set differs.

Numeric missing values are median-imputed, with missingness indicators; numeric
scaling statistics and one-hot categories are fitted from the 515 training
rows only using the existing Logistic Regression preprocessor. Test-only
categories map to zero-valued columns. No preprocessing is fitted on test data.
Features, IDs, scores, targets, and label mapping are recorded in the versioned
metadata. Training is reproducible with a fixed seed and one worker.

The initial fixed configuration uses `multi:softprob` for class probabilities,
100 trees, depth 2, learning rate 0.05, minimum child weight 5, 0.8 row and
column subsampling, L2 regularization 2, histogram tree construction, and no
hyperparameter search. XGBoost's float32 probability outputs are checked for
finite values in [0, 1] and sums within 1e-6 of one.

Run from the repository root with `py -m ml.train_xgboost`. The model JSON,
training metadata, and preprocessing artifact are saved under
`ml/models/xgboost/v1/`. Evaluation JSON, per-match test predictions, and the
direct comparison report are saved under `ml/evaluation/xgboost/v1/`.

### v1 chronological test results

All rows below are from the same 143 matches (2026-10-31–2026-11-21):

| Model | Log Loss | Brier | Accuracy |
|---|---:|---:|---:|
| Poisson v1 | 0.9719 | 0.5780 | 51.05% |
| Logistic Regression v1 | 1.1635 | 0.6869 | 41.26% |
| XGBoost v1 | 1.0186 | 0.6079 | 49.65% |

Training used 515 matches from 2025-08-15 through 2026-10-30. XGBoost is better
than Logistic Regression on the three aggregate metrics, but Poisson remains
best on log loss, Brier score, and accuracy. XGBoost's top-class expected
calibration error is 0.0501 (mean confidence 0.4879; accuracy 0.4965); this is
descriptive for a small holdout and does not establish reliable calibration.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Home win | 0.516 | 0.738 | 0.608 | 65 |
| Draw | 0.143 | 0.036 | 0.057 | 28 |
| Away win | 0.512 | 0.440 | 0.473 | 50 |

Confusion matrix (actual rows, predicted columns; order home win, draw, away
win):

```text
[[48,  6, 11],
 [17,  1, 10],
 [28,  0, 22]]
```

The model identifies few draws on this test period. The holdout is small, the
configuration is a baseline rather than a tuned model, and provider-limited
historical data constrains both training size and generalization. It is not
production-ready; do not promote it based on accuracy or this single evaluation.
