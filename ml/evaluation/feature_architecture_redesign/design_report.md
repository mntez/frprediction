# FRPRECTIONS Feature Architecture Redesign — Design Audit

**Status:** Audit and design only; no code, database, dataset, or model artifact was changed.  
**Dataset inspected:** `ml/data/processed/training_dataset.csv`  
**Dataset SHA-256:** `862e67cdf4f8d3438f2e0fcc877acfcea7a50f22a120d69a3062e224bf06f5cc`

## Executive summary

The expanded CSV has 2,700 eligible matches, 39 match-information/pre-match columns, and 10 targets. Historical coverage materially improved feature completeness: rolling form/goal averages are missing for about 2.4% of rows (previously about 9%), and venue rates for 4.9% (previously about 17.8%). All four head-to-head fields remain completely missing, so they still provide no signal.

The current feature table contains high redundancy. Prior played counts and same-venue played counts have Spearman correlation 0.998; wins and cumulative goals scored correlate around 0.95; exposure counts and cumulative goals conceded correlate above 0.92. Several feature/target correlations are plausible but modest; the strongest univariate training-period association is about 0.20. Draws are particularly weakly associated with individual numeric inputs.

Recommended compact candidate: four normalized, shrunk team attack/defence strengths; six numeric five-match form/goals summaries; and one pre-match Elo difference. Treat four venue-residual features as a separate ablation because they overlap with team strength. Do not carry forward the all-null H2H fields or raw categorical form strings. Keep the saved Poisson v1 benchmark and its test predictions immutable.

## 1. Expanded dataset analysis

### Coverage and time split

- 2,700 rows, 49 total columns: 39 feature/match-information columns and 10 targets.
- Match dates: 2025-08-08 through 2026-11-21.
- Split at the existing cutoff, 2026-10-31: 2,557 current training-period rows and 143 test rows.
- The 143 ordered test fixture IDs match the saved Poisson v1, Logistic Regression v1, and XGBoost v1 prediction files exactly. The saved Poisson test ID order has SHA-256 `88e9d715868dad81f0ce47cb47f6afbdba0585a08b5dbbc2f4f432f97ac36f92`; the exact IDs are in `ml/evaluation/poisson_baseline/v1/test_predictions.csv`.
- Current expanded training IDs, ordered by date and provider ID, have SHA-256 `5b171d12b0f6223c26f415670cf1e21fce47c7d4e11e3525600ea484a196befb`.

The old Poisson v1 evaluation records 515 training rows, 143 test rows, and source dataset hash `fef1f4a64294baf6fe93c833f13ab15bc0409e05d058c7520073ef2d58a1a718`. Its exact 515 training fixture IDs were not saved as a manifest, and the old CSV was replaced during the approved historical expansion. Therefore the exact original training ID list cannot now be independently reconstructed from the checked-in artifacts. The test fixture list is preserved. Future model artifacts should store both ordered training and test IDs, or their explicit manifests.

The v1 benchmark is still the best recorded model: Log Loss 0.9719, Brier 0.5780, accuracy 51.05%, goal MAE 0.9605, and goal RMSE 1.2606. It was fit on the earlier 515-row dataset. The expanded dataset changes the pre-match history available to later matches, so a v2 trained on the expanded rows would compare the combined effect of more training data and redesigned features against v1. Preserve v1, and add a separately versioned Poisson comparator trained with the expanded history if the goal is to isolate feature gains.

### Missing values and feature types

Missing percentages below use all 2,700 rows. “Identifiers/date” are kept for joins, chronology, or audit and are excluded from all three current model inputs.

| Current columns | Type | Missing rate | Current model usage |
|---|---|---:|---|
| `provider_match_id`, `season_provider_id`, `home_team_provider_id`, `away_team_provider_id` | Identifier | 0% | None; excluded |
| `match_date` | Date | 0% | None; excluded |
| `competition`, `home_team`, `away_team` | Categorical | 0% | Logistic Regression and XGBoost |
| `season_start_year` | Numeric/time proxy | 0% | Logistic Regression and XGBoost |
| Home/away prior match, win, draw, loss, goals-for and goals-against counts | Numeric | 0% | Logistic Regression and XGBoost |
| `home_home_matches_played`, `away_away_matches_played` | Numeric | 0% | All models; part of Poisson's six inputs |
| Home rolling goals-for/goals-against and `home_recent_form` | Numeric/numeric/categorical | 2.48% each (67 rows) | Logistic Regression and XGBoost |
| Away rolling goals-for/goals-against and `away_recent_form` | Numeric/numeric/categorical | 2.37% each (64 rows) | Logistic Regression and XGBoost |
| Home home-win rate and home goals-for/goals-against averages | Numeric | 4.85% each (131 rows) | Logistic Regression and XGBoost; both home goal averages also Poisson |
| Away away-win rate and away goals-for/goals-against averages | Numeric | 4.85% each (131 rows) | Logistic Regression and XGBoost; both away goal averages also Poisson |
| Four `h2h_*` columns | Numeric | 100% (2,700 rows) | Listed by Logistic Regression/XGBoost, but constant after training imputation; no usable signal |

Targets have no missing values. The feature builder calculates history only from matches before the target's UTC date; all matches on one UTC date are featurized before any result from that date enters history. It includes only `FINISHED`/`AWARDED` records with results and excludes cancelled/in-play records.

### Feature distributions and redundancy

- Prior appearances per team: median 22, 90th percentile 41, maximum 50. Same-venue appearances: median about 11, 90th percentile 20, maximum 25.
- Cumulative goals-for has median 28 and 95th percentile about 75; cumulative goals-against has similar exposure-driven ranges. These totals are strongly dependent on how many prior matches a team has played.
- Recent five-match goals averages have median 1.4 goals per match; their 95th percentiles are 2.6–2.8. The range includes zero-goal runs and maximum five/six-goal match averages.
- Home team's home win-rate median is 0.438; away team's away win-rate median is 0.273. These are estimated from fewer same-venue matches than overall form.
- `season_start_year` contains only 2025 and 2026. Its apparent signal can reflect season/league mix or temporal changes rather than a stable team property.
- There are 131 home-team and 131 away-team name categories. Recent-form strings have 320 distinct non-missing categories in the current training rows for each side. With the existing numeric scaling/missing-indicator and one-hot setup, the expanded training feature matrix would have approximately 971 columns. Sparse one-hot form sequences are an unnecessarily high-dimensional representation of five outcomes.

Largest pairwise Spearman correlations among numeric predictors, calculated on the 2,557 pre-cutoff rows, include:

| Feature pair | Spearman rho | Interpretation |
|---|---:|---|
| Home prior matches / home prior home matches | 0.998 | Nearly the same exposure ranking |
| Away prior matches / away prior away matches | 0.998 | Nearly the same exposure ranking |
| Home wins / home cumulative goals scored | 0.949 | Team strength and number of appearances are entangled |
| Away wins / away cumulative goals scored | 0.948 | Same exposure/strength overlap |
| Home prior matches / home goals conceded | 0.927 | Cumulative goals largely encode history length |
| Away prior matches / away goals conceded | 0.926 | Same exposure/strength overlap |
| Home home-win rate / home home-goals-scored average | 0.741 | Related venue performance summaries |

The identity `wins + draws + losses = prior matches` is also structural; one outcome count is redundant when the other two and total count are present. Numeric scaling in the existing Logistic Regression/XGBoost preprocessor is fitted on training rows only. One-hot categories are also training-fitted, with missing values represented explicitly and unseen test categories all-zero. Scaling is appropriate for Logistic Regression; XGBoost trees do not require standardized numeric columns, so model-specific preprocessing can avoid needless work in a future pipeline.

### Competition differences and target balance

Observed goal rates vary across competitions in the expanded dataset:

| Competition | Matches | Home goals / match | Away goals / match | Total goals / match |
|---|---:|---:|---:|---:|
| BL1 | 394 | 1.835 | 1.431 | 3.266 |
| DED | 419 | 1.809 | 1.568 | 3.377 |
| FL1 | 402 | 1.587 | 1.259 | 2.846 |
| PD | 504 | 1.579 | 1.157 | 2.736 |
| PL | 487 | 1.528 | 1.220 | 2.747 |
| SA | 494 | 1.352 | 1.231 | 2.583 |

Overall targets: home win 1,183 (43.8%), draw 678 (25.1%), away win 839 (31.1%); Over 1.5 2,123 (78.6%), Over 2.5 1,499 (55.5%), Over 3.5 892 (33.0%), BTTS 1,512 (56.0%). League differences are meaningful: DED total-goals average is 3.377 versus 2.583 in SA; Over 2.5 rates range from 64.9% in DED to 49.0% in SA. Competition context or within-competition normalization should remain available.

| Competition | Home win | Draw | Away win | Over 2.5 |
|---|---:|---:|---:|---:|
| BL1 | 44.9% | 23.9% | 31.2% | 64.5% |
| DED | 41.3% | 26.3% | 32.5% | 64.9% |
| FL1 | 45.8% | 24.1% | 30.1% | 53.2% |
| PD | 47.6% | 24.8% | 27.6% | 51.2% |
| PL | 42.5% | 27.5% | 30.0% | 53.2% |
| SA | 40.9% | 23.9% | 35.2% | 49.0% |

### Does expanded history improve reliability and signal?

Yes for feature availability: rolling-history missingness fell from about 9% to 2.4%, and venue-rate missingness fell from about 17.8% to 4.9%. The four H2H features remain 100% missing because no target row has three prior meetings for that team pair under current rules. Do not keep those columns in the model until they have nonconstant coverage.

Univariate Spearman associations between numeric pre-match features and outcomes, using training rows only, remain modest. The strongest absolute associations are home win vs `home_home_win_rate` (rho +0.203), away win vs that same rate (-0.165), Over 2.5 vs `home_home_goals_scored_avg` (+0.106), and Over 3.5 vs the same feature (+0.099). The strongest draw association is only +0.066. These are exploratory, unadjusted correlations with correlated inputs and multiple comparisons; they do not demonstrate incremental predictive value or causality. No feature selection was based on the final 143 test outcomes.

**Recommendation on existing columns:** remove the four all-null H2H features for the next dataset schema; exclude all provider IDs and raw match date from predictors as already done; replace recent-form strings, cumulative W/D/L counts, and cumulative goal totals with compact numeric rates; retain team names only as a controlled identity-feature ablation; and test whether `season_start_year` adds value before keeping it. Preserve competition as categorical context. Keep at most one history-exposure control per side if validation shows it helps.

## 2. Proposed compact feature architecture

All proposed history is restricted to prior completed matches in the **same competition** through the previous UTC date. A pre-match, expanding competition scoring baseline is calculated from only records available before each feature timestamp. For a prior match `i`, let `mu_for_i` be that competition's expected goals for a team in its home/away venue, and `mu_against_i` be the corresponding expected goals conceded. Let `w_i = 2^(-age_days_i / 365)` initially, and `kappa = 5` neutral-prior matches. These recency/shrinkage values are starting hypotheses to validate, not fixed truths.

For team `t`, define `r_i = goals_for_i / mu_for_i` and `c_i = goals_against_i / mu_against_i`. Then:

```text
attack(t)  = (kappa + sum_i(w_i * r_i)) / (kappa + sum_i(w_i))
defense(t) = (kappa + sum_i(w_i)) / (kappa + sum_i(w_i * c_i))
```

Both are dimensionless and neutral at 1. Higher attack means more scoring than the competition/venue baseline; higher defense means fewer goals conceded than expected. For no team history, the neutral prior value 1 is an explicit prior, not a fabricated match statistic. If the competition baseline itself is not yet estimable, return missing and use training-fitted imputation plus a missing indicator. An implementation may also retain one effective prior-match count per side as a reliability ablation.

### A. Team strength — core candidate

| Feature | Definition, history and missing behavior | Normalization, leakage control and rationale |
|---|---|---|
| `home_attack_strength` | `attack(home_team)` from all prior same-competition matches, adjusted by each match's home/away scoring baseline. Shrink toward 1 with `kappa=5`; no prior team history yields 1. | Already relative to the competition scoring environment; no extra normalization needed for a tree. Use training-fold scaling for Logistic Regression. Only prior UTC dates enter. Captures sustainable scoring ability without cumulative-goal exposure. |
| `home_defense_strength` | `defense(home_team)` from the same history, with goals against normalized by the baseline expected for the opponent's scoring venue. Shrink toward 1; no history yields 1. | Dimensionless with higher values meaning stronger defence. Same time boundary. Captures goals prevented while separating defence from raw team match count. |
| `away_attack_strength` | `attack(away_team)` from the away club's prior same-competition matches at either venue. | Same normalization and chronology. Estimates away side scoring ability, beyond a single away-only average. |
| `away_defense_strength` | `defense(away_team)` from the away club's prior same-competition matches at either venue. | Same normalization and chronology. Estimates away side goals prevention. |

These are the recommended core team-strength features. They may still correlate with form; assess incremental value through group ablation rather than keeping both families automatically.

### B. Recent form — core candidate

Use the team's last `min(5, n)` prior same-competition matches across seasons, ordered by UTC kickoff. Define points as 3 for a win, 1 for a draw, and 0 for a loss. Use per-match averages so partial histories are not mechanically smaller than full five-match histories.

| Feature | Exact definition | Missing, normalization and leakage |
|---|---|---|
| `home_form_points_5` | Sum of points in the home team's prior five matches divided by the number observed (range 0–3). | Missing if no prior match; impute with training-only median plus missing indicator. Logistic Regression may scale; tree models need not. All rows earlier than the target UTC date. |
| `away_form_points_5` | Same calculation for the away team. | Same policy. |
| `home_goals_for_5` | Mean home-team goals scored over those same prior up-to-five matches. | Missing only if no history; do not fill unavailable match slots with zero. Scale for Logistic Regression only. Earlier-date results only. |
| `home_goals_against_5` | Mean goals conceded by the home team over those matches. | Same policy. |
| `away_goals_for_5` | Mean away-team goals scored over those matches. | Same policy. |
| `away_goals_against_5` | Mean goals conceded by the away team over those matches. | Same policy. |

The existing categorical `WWD...` strings encode the same five outcomes with hundreds of sparse categories. Replace them with points per match; the rolling goal means add score-intensity information not captured by outcome points.

### C. Venue performance — optional ablation, not in first core set

Calculate `attack_venue(t,v)` and `defense_venue(t,v)` with the same formulas as group A but only using prior appearances at venue `v`, with the same recency weights and shrinkage. Define venue residuals as:

```text
venue_attack(t,v)  = log(attack_venue(t,v)) - log(attack(t))
venue_defense(t,v) = log(defense_venue(t,v)) - log(defense(t))
```

Positive values mean the team performs better in that venue role than its overall strength estimate. Log ratios are centered around zero and dimensionless. These are undefined if either baseline or the required team history is unavailable; retain missingness indicators rather than substituting fabricated observations. All calculations use only prior UTC dates; the model preprocessor handles imputation from training data.

| Feature | Team/venue mapping | Why it may help / redundancy risk |
|---|---|---|
| `home_venue_attack` | Home team, home appearances | Captures home-specific scoring beyond overall attack; partly duplicates home attack and current Poisson input. |
| `home_venue_defense` | Home team, home appearances | Captures home-specific concession rate; partly duplicates overall defence and current Poisson input. |
| `away_venue_attack` | Away team, away appearances | Captures away-specific scoring; useful if away performance differs from overall attack. |
| `away_venue_defense` | Away team, away appearances | Captures away-specific concession rate. |

Use these four as a separate ablation. Add them only if rolling-origin validation improves probability and goal metrics consistently. The residual definition is preferred to adding a second set of raw venue averages because it expresses only the venue-specific deviation from group A.

### D. Elo — one derived difference in the compact feature set

| Proposed feature | Definition, history and missing behavior | Normalization, leakage control and rationale |
|---|---|---|
| `home_team_elo` | Home team's competition-scoped rating immediately before the target UTC date. New/unrated team prior: 1500. | Rating points use a 400-point scale; Logistic Regression can scale using training data. Rating is generated forward only. Captures longer-run result strength. |
| `away_team_elo` | Away team's competition-scoped rating immediately before the target UTC date. New/unrated team prior: 1500. | Same scale and chronology. |
| `elo_difference` | `home_team_elo - away_team_elo`; home advantage is handled separately in the Elo update expectation and by the match model/competition context. | Same-competition difference only; do not compare absolute ratings between independently initialized leagues. This is the preferred compact Elo feature. |

Do not include all three in the initial model: `elo_difference` is exactly determined by the two rating columns. Start with `elo_difference`; keep absolute ratings only for diagnostic or ablation purposes. Elo may summarize result-based team strength on a dynamic time scale, but its incremental value over group A and B must be established by chronological validation.

**Recommended first compact candidate:** groups A and B plus `elo_difference` (11 features). Evaluate group C separately (15 total if all four are added). Do not include H2H, team IDs as numeric values, raw recent-form strings, or cumulative goal totals in this first candidate. Team-name identity can be tested as a separate ablation.

## 3. Elo methodology and safeguards

Research on association football has used pre-match Elo differences as covariates in match-outcome models and updated team ratings after matches; the cited work tunes Elo parameters using training data and treats home advantage in the outcome model. Its supplementary material describes dynamic pre-match ratings and a goal-margin adjustment option ([Hvattum & Arntzen, 2010](https://www.sciencedirect.com/science/article/pii/S0169207009001708); [Arntzen & Hvattum supplementary material](https://statmod.org/smij/Vol21/Iss5/Arntzen/Supplement.pdf)). A separate Elo-MOV study notes that draws need an explicit ternary outcome model when Elo is used to produce 1X2 probabilities, rather than treating a binary win expectation as three-class probabilities ([Kiraly & Qian, G-Elo](https://arxiv.org/abs/2010.11187)). Here Elo is proposed as a feature, not as a standalone 1X2 probability generator.

### Starting design

Maintain separate rating pools per competition because these data contain no cross-league matches that would identify absolute strength differences between the six leagues. Keep competition as a model context feature. A team's absolute rating is meaningful within its competition pool; the within-match rating difference is the candidate feature.

For a match on a given UTC date, read both ratings before any result from that date is processed:

```text
E_home = 1 / (1 + 10^(-((R_home_before - R_away_before + H) / 400)))
S_home = 1.0 for home win, 0.5 for draw, 0.0 for away win
delta  = K * (S_home - E_home)
R_home_after = R_home_before + delta
R_away_after = R_away_before - delta
```

`home_team_elo` and `away_team_elo` store the before-state; `elo_difference` is computed from those before-state values. Compute features for all matches on a UTC date first, then apply updates for that date. If a team appears twice on one UTC date, neither match sees the other's result. Store or derive before- and after-ratings separately; never mutate the pre-match snapshot after the result is known.

| Parameter | Initial proposal | Validation |
|---|---:|---|
| Initial rating | 1500 within each competition | Verify that a common offset does not affect differences; test new-team behavior. |
| Elo scale | 400 points | Keep fixed initially; changing scale is equivalent to changing the rating sensitivity. |
| K factor | 10 | Compare a small predeclared grid such as 10 and 20 on rolling-origin validation only. Football studies have tuned K from training data; published values are not assumed optimal here. |
| Home advantage `H` | 60 Elo points in the expected-score update | Compare 0, 40, 60, and 80 on training/validation periods; do not tune on the final 143. |
| Margin-of-victory multiplier | Off for initial version (`M=1`) | Optionally test `M=(1+|goal_diff|)^b`, with `b=1`, as a separate validation ablation. Start without it to limit sensitivity to score outliers and the awarded-score issue. |
| Off-season regression | `R_start = 1500 + 0.75 * (R_end - 1500)` for returning teams | Compare no regression (`rho=1`) with 0.75 using earlier seasons only. New teams start at 1500. |

### Promotion, relegation and competition boundaries

The current data covers only the six top divisions, not lower-division match histories. A team first appearing in a competition therefore starts at that competition's neutral 1500 prior; do not infer a promoted team's strength from a league that is absent from the database. A returning team retains its competition-specific rating, with optional validated offseason regression. If a club is absent for a season and returns, apply the same recorded regression policy. A club changing competition starts from the new competition's prior unless the data later adds a validated bridge between rating pools. Ratings from different competitions must not be directly compared.

The 2025/26 dataset is the first full season in the expanded history. Its opening matches necessarily have neutral or weakly informed Elo values because earlier seasons are not present. Avoid a retrospective initialization computed from the entire 2025/26 season: that would use future match results for early-season rows. Initialize prospectively at 1500 and update only as historical results become available.

**Awarded matches require an explicit policy.** The current dataset accepts `AWARDED` and derives targets from stored full-time scores. Before using an awarded 0–0 score as a draw update, confirm the provider's official winner/result semantics. If the canonical sporting result is not represented by those scores, exclude the fixture from Elo updates (and resolve target handling in a separately approved data-quality change); never convert `AWARDED` to an ordinary `FINISHED` match.

## 4. Leakage and missing-data rules

1. Construct all historical rates, form, venue residuals, and Elo states from results with kickoff UTC date strictly earlier than the target's UTC date, retaining the existing same-day batch rule.
2. For Elo, emit `*_elo_before`/`elo_difference` before applying any result. Apply rating changes only after the date batch is fully featurized.
3. Competition baselines must be expanding/as-of baselines; do not calculate a full-season league average and backfill it into early season rows.
4. Keep provider IDs, dates, scores, target columns, and post-match result fields out of model features. Target values remain derived from actual stored scores and must match them.
5. If there are no team observations, a strength value of 1 or rating 1500 is an explicit neutral prior, not a claimed observation. Keep an exposure/missing indicator so models can distinguish a neutral estimate with no evidence from an estimate supported by matches.
6. For undefined venue/form features, leave missing. Fit imputation statistics and category vocabularies on each training fold only. Use numeric missing indicators; represent categorical missing explicitly; map unseen categories without looking at validation/test categories.
7. Do not impute missing score history as zero or invent matches/statistics. Require a documented minimum history for competition baselines; until it is met, emit missing values and an indicator.
8. Feature output must be deterministic given the same ordered source records and feature configuration.

## 5. Evaluation design

### Preserve v1 and lock the final test fixtures

Keep the 143 saved Poisson v1 test fixture IDs exactly as recorded in `ml/evaluation/poisson_baseline/v1/test_predictions.csv` (cutoff 2026-10-31, dates 2026-10-31 through 2026-11-21). The expanded CSV's chronological split reproduces that ordered test ID list exactly; Logistic Regression and XGBoost prediction files do as well. Do not use outcomes in this period for feature selection, Elo parameter tuning, or model choice.

Poisson v1's 515 training fixture IDs are not persisted, so the old exact training sample cannot be recovered from the present repository. The expanded dataset has 2,557 rows before the same cutoff. Preserve Poisson v1 metrics/artifacts as the historical benchmark; for a controlled feature comparison, add a new expanded-history Poisson comparator trained on the expanded pre-cutoff rows with its own versioned artifacts and the same final test IDs. Label the difference between that comparator and v1 as involving both training-data coverage and its refreshed inputs.

### Rolling-origin validation before the final test

Use expanding chronological folds wholly before 2026-10-31. Example windows on the current dataset:

| Fold | Train dates | Train rows | Validation dates | Validation rows |
|---|---|---:|---|---:|
| 1 | 2025-08-08–2025-12-31 | 954 | 2026-01-01–2026-03-31 | 682 |
| 2 | 2025-08-08–2026-03-31 | 1,636 | 2026-04-01–2026-06-30 | 422 |
| 3 | 2025-08-08–2026-06-30 | 2,058 | 2026-08-01–2026-09-30 | 313 |
| 4 | 2025-08-08–2026-09-30 | 2,371 | 2026-10-01–2026-10-30 | 186 |

The fold windows are examples to preregister before implementing/tuning. Keep same-date fixtures together. Fit imputation, scaling, categorical encoders, and any learned baselines on each fold's training rows only. Tune feature groups and a small Elo parameter grid on the rolling folds; select using average probability quality and fold stability, not one favorable period. If more seasons are later added, include full-season holdouts.

After feature and parameter decisions are frozen, evaluate once on the original 143 fixtures. Report:

- 1X2 Log Loss, multiclass Brier Score, accuracy, per-class precision/recall/F1, and calibration (reliability bins/ECE).
- Goal MAE and RMSE for home goals, away goals, and pooled goals; retain total-goals metrics where supported.
- Market calibration and scores for Over 1.5/2.5/3.5 and BTTS if those outputs are produced.
- Exact training/validation/test fixture ID manifests, dataset/config hashes, date ranges, and feature-schema version.

The final set is only 143 matches over 22 days, so its metrics have high uncertainty. Preserve the set anyway for comparability; add newer, strictly future test periods later rather than replacing it.

## 6. Proposed implementation sequence

1. Freeze a v2 feature schema and definitions above; decide awarded-result semantics and whether the competition-scoring baseline has enough pre-match observations.
2. Add a chronological state builder for strength, rolling form, venue residuals, and Elo-before features. Keep the current v1 feature builder intact or version its output so v1 artifacts remain reproducible.
3. Write the v2 CSV to a separate path (for example `ml/data/processed/training_dataset_v2.csv`) with stable columns, source hash, feature version, and ordered fixture IDs. Do not overwrite the expanded v1 CSV until an explicit migration decision.
4. Add unit tests for formulas, shrinkage, missing histories, competition boundaries, deterministic processing, and all leakage invariants.
5. Run feature-group ablations on the rolling-origin folds. Start with A+B+Elo difference; test venue residuals and team identity separately. Do not inspect the final test metrics to choose features.
6. Create new model/evaluation artifact directories and compare against both preserved Poisson v1 and the expanded-history Poisson comparator. Do not overwrite any existing v1 artifact.
7. Document the final schema, data version, Elo parameters, splits, metrics, and limitations in `ml/README.md`.

## 7. Likely files and tests

No implementation was made in this audit. Likely future changes:

| File/path | Expected responsibility |
|---|---|
| `ml/features/feature_engineering.py` | Add versioned compact features and chronological team/Elo state construction. |
| `backend/app/jobs/build_training_dataset.py` | Add a versioned output/schema path and stable feature metadata; continue reading only eligible raw results. |
| `ml/baselines/logistic_regression.py` | Add v2 feature configuration or a versioned preprocessor without changing v1 behavior. |
| `ml/baselines/poisson.py` or a new versioned Poisson module | Keep v1 estimator intact; implement an expanded-history comparator using v2 strength inputs if selected. |
| New `ml/train_*_v2.py` scripts and `ml/models/.../v2/`, `ml/evaluation/.../v2/` | Train/evaluate versioned candidates and persist ID manifests/metrics without overwriting v1. |
| `backend/tests/test_feature_engineering.py` and new Elo tests | Assert formulas, UTC same-day batching, no own-result/future leakage, first-match priors, club/competition boundaries, season regression, awarded handling, and deterministic output. |
| `backend/tests/test_logistic_regression.py`, `test_poisson_baseline.py`, `test_xgboost.py` | Assert train-only preprocessing, fixed chronological IDs, valid probabilities, reproducible evaluation, and no changes to v1 artifact paths. |
| `ml/README.md` | Document v2 schema, feature formulas, missing-data behavior, Elo policy, folds, and results. |

No database schema change or frontend change is required if the proposed features are generated in the offline processed dataset. No new dependency is indicated.

## Addendum — Final design clarifications (2026-11-22)

Audit only: no code, PostgreSQL, processed data, or model artifacts were changed.

### Controlled Poisson comparisons

| Benchmark | Specification and interpretation |
|---|---|
| **A — Original Poisson v1** | Preserve `ml/models/poisson_baseline/v1/` and `ml/evaluation/poisson_baseline/v1/` unchanged, including its metrics (Log Loss 0.9719, Brier 0.5780, accuracy 51.05%, goal MAE 0.9605, RMSE 1.2606) and ordered 143 test fixture IDs. The original 515 training IDs were not saved, so that sample cannot be reconstructed exactly. |
| **B — Expanded-history comparator** | Reuse the original `PoissonBaseline` estimator and exactly its six inputs: `home_home_matches_played`, `home_home_goals_scored_avg`, `home_home_goals_conceded_avg`, `away_away_matches_played`, `away_away_goals_scored_avg`, `away_away_goals_conceded_avg`. Fit on the 2,557 current rows before 2026-10-31 and evaluate on the same 143 IDs as A. Save under `ml/models/poisson_baseline/expanded_history/v1/` and `ml/evaluation/poisson_baseline/expanded_history/v1/`. B versus A is the closest estimate of expanded-history impact, but combines more training rows with refreshed values for those six pre-match inputs; it is not a pure sample-size experiment. |
| **C — Redesigned Poisson v2** | Use the 11 core features (four normalized strengths, six recent-form/goals summaries, Elo difference) plus competition context, with the same test IDs. The proposed target-clean training count is 2,556 (see awarded-fixture conflict below); the request's 2,557 count cannot be preserved while excluding that fixture. Fit separate regularized Poisson log-link home-goal and away-goal models; derive score/market probabilities from their independent Poisson grid. Fit coefficients and preprocessing on training data only; select regularization on the predeclared rolling folds. Save under `ml/models/poisson_baseline/v2/` and `ml/evaluation/poisson_baseline/v2/`. B versus C compares the full new feature/modeling package, so cannot attribute a metric change to features alone. Use rolling-fold feature ablations for incremental value. |

Persist ordered train/test ID manifests, dataset/config hashes, and per-match predictions for B and C. Keep all test IDs identical. Earlier test results may update *state* for later test dates once available, but never refit model parameters or preprocessing. A versus C is an overall benchmark comparison only.

### Rolling-origin state and fitted transformations

For each fold, fit imputers, missingness indicators, scalers, category encoders, Poisson coefficients, and regularization choices on that fold's training rows only; transform validation rows with those fixed objects. These fitted objects do not change inside validation.

Competition scoring baselines, normalized strengths, form, venue residuals, and Elo are chronological historical state, not fold-fitted transformations. Construct features date by date in UTC. For every UTC date, featurize all its fixtures from the state before that date; update the state only after that date's eligible outcomes are available. Thus earlier completed validation matches can inform later validation dates, but no result can affect its own row or another fixture on the same UTC date. Validation outcomes never refit the model. Apply the same walk-forward rule to the locked test period.

Baselines update from prior competition matches; strength and venue histories use each past appearance's then-available baseline; form uses each team's last five prior same-competition appearances; Elo uses opening-of-date ratings and date-batched updates. Exclude the awarded fixture below from v2 states until its timing and eligibility are represented.

### FL1 AWARDED fixture 542704

- Database row: FL1 season provider ID 2420; FC Nantes (team ID 543) vs Toulouse FC (511), matchday 34; 2026-05-17 19:00 UTC (22:00 East Africa Time); `AWARDED`; 0–0; winner `DRAW`. Fixture and result `provider_metadata` are both null.
- The live provider response contains `id=542704`, `utcDate=2026-05-17T19:00:00Z`, `status=AWARDED`, `stage=REGULAR_SEASON`, `matchday=34`, competition FL1/Ligue 1, season 2420 (2025-08-17–2026-05-16), both team IDs/names, `score.duration=REGULAR`, `score.winner=DRAW`, `score.fullTime={home:0,away:0}`, null half-time, regular-time, extra-time and penalty scores, and `lastUpdated=2026-05-30T20:20:41Z`. It also returns referee and odds-access fields, but no match statistics or result-availability timestamp. Ingestion retained status, full-time score and winner, not the original response payload.
- The current builder includes `AWARDED`; it derives `draw=1`, 0 total goals, all overs 0 and BTTS 0 directly from the score. Ligue 1 reported the match was stopped shortly after minute 20 for safety ([Ligue 1 report, 17 May 2026](https://ligue1.com/fr/articles/l1_article_5119-j34-nantes-toulouse-definitivement-arrete)); FC Nantes later reported the LFP commission ratified 0–0 ([club notice, 27 May 2026](https://www.fcnantes.com/articles/article2809.php?num=49229)). The official league resolution is therefore a draw at 0–0, but not a completed 90-minute sporting performance. The result was ratified after kickoff; neither database nor training row records when it became available.
- **Recommendation:** exclude this fixture from v2 training and all team scoring/Elo state for now. Its official league draw is known, but current records cannot model its administrative status and availability time safely; its 0–0 is not a normal full-match goal observation. The fixture is one of the 2,557 pre-cutoff rows, so this makes the eligible v2 goal-training set 2,556 rows. This conflicts with the requirement that both expanded-history B and v2 C train on exactly 2,557 rows. Preserve B's explicitly requested 2,557-row comparator, but do not claim B/C is a strictly matched training-set comparison unless the row policy is resolved. Recommended clean comparison is to refit B on the same 2,556 eligible IDs as C; this changes B's requested row count by one. Do not update Elo at kickoff; absent an explicit availability time, exclude it. The uncertainty is modeling eligibility/timing, not the ratified score.

### Final feature defaults

**Competition baselines.** For competition `c` and UTC date `d`, use only prior eligible `FINISHED` matches in `c` with UTC date `< d`:

```text
mu_home(c,d) = prior home-team goals / prior home fixtures
mu_away(c,d) = prior away-team goals / prior away fixtures
```

Each venue mean is available only after **20 observations in that venue role**. Before then, output missing with a training-fitted missing indicator; do not substitute a full-season, other-competition, or future mean. Update after the date batch. When normalizing any past team appearance, use its own pre-date baseline; if that baseline was unavailable, omit the appearance from normalized strength calculations rather than retrospectively applying a later mean.

Retain the proposed 365-day half-life `w_i=2^(-age_days_i/365)` and `kappa=5`. For an eligible appearance, `r_i=goals_for_i/mu_venue_i` and `c_i=goals_against_i/mu_opponent_venue_i`:

```text
attack  = (5 + sum(w_i*r_i)) / (5 + sum(w_i))
defence = (5 + sum(w_i))   / (5 + sum(w_i*c_i))
```

If a team has no usable history but its competition baseline exists, use neutral strength 1; if the baseline is unavailable, strength is missing. Use `FINISHED` records only until awarded handling is approved.

**Venue residuals.** Require at least **5 prior same-competition appearances in that venue role** and available overall strengths/baselines. Apply the same weights and `kappa=5` to venue-only estimates, then compute `log(venue_attack)-log(overall_attack)` and `log(venue_defence)-log(overall_defence)`. Below five, emit missing, not a fabricated neutral observation. The first residual-only ablation uses fold-fitted missing indicators; test venue counts separately rather than bundling them into the four residuals.

**History exposure.** Keep total prior-match count out of the 11-feature core. Test exactly home and away prior eligible same-competition match counts as a separate ablation: they may convey estimate reliability, but overlap strongly with venue counts. Retain only if gains are stable across rolling folds.

**Elo.** Use separate competition pools, new-team rating 1500, scale 400, `K=10`, home advantage `H=60`, and no margin multiplier. At a new season, returning teams regress to `1500 + 0.75*(previous_end-1500)`; apply this once per elapsed season transition. A team new to or moving into a competition starts at 1500 in that pool; never transfer ratings between leagues. Use `elo_difference = home_rating_before_date - away_rating_before_date`. Compute same-date deltas from opening ratings and aggregate per team before applying them. Only eligible `FINISHED` results update Elo; exclude `AWARDED` until its availability policy is explicit.

### Revised implementation sequence

1. Freeze these formulas and eligibility rules; keep v1 immutable and exclude AWARDED from v2 until status/availability can be represented. Resolve the B/C training-count conflict (2,557 requested versus 2,556 eligible) before training.
2. Add a versioned point-in-time state builder and separate v2 dataset; do not overwrite the current dataset or builder outputs.
3. Test formulas, UTC date batching, historical-baseline availability, missing histories, season transitions, competition boundaries, awarded exclusion, and own/future leakage.
4. Verify v2 row IDs/schema/missingness, then run B followed by C and predeclared venue/exposure ablations on rolling folds only.
5. Freeze choices before evaluating once on the original 143 test IDs. Save manifests, hashes, predictions and metrics to new artifact paths; document results in `ml/README.md`.

## 8. References

- Hvattum, L. M. & Arntzen, H. (2010), [Using ELO ratings for match result prediction in association football](https://www.sciencedirect.com/science/article/pii/S0169207009001708), *International Journal of Forecasting*.
- Arntzen & Hvattum, [supplementary material on dynamic team ratings](https://statmod.org/smij/Vol21/Iss5/Arntzen/Supplement.pdf).
- Kiraly & Qian, [Elo-MOV: Generalization of the Elo algorithm by modelling the discretized margin of victory](https://arxiv.org/abs/2010.11187), including the distinction between an Elo expected score and explicit three-outcome probabilities.
