# Dataset and feature quality analysis

**Analysis date:** 2026-11-22  
**Scope:** Read-only review of PostgreSQL, the generated CSV, current feature rules, and saved baseline evaluations. No raw rows, feature rules, or model artifacts were modified.

## 1. Historical data coverage

PostgreSQL contains **6 competitions, 7 season records, 117 teams, 2,074 fixtures, and 661 result rows**. “Completed matches” below counts results attached to `FINISHED` or `AWARDED` fixtures; the database currently has no `AWARDED` fixtures.

| Competition | Stored season | Completed matches |
|---|---:|---:|
| BL1 | 2026/27 | 88 |
| DED | 2026/27 | 113 |
| FL1 | 2026/27 | 96 |
| PD | 2026/27 | 124 |
| PL | 2025/26 | 16 |
| PL | 2026/27 | 107 |
| SA | 2026/27 | 114 |
| **Total** | **7 season records** | **658** |

The processed training CSV has **658 completed-match rows**, dated 2025-08-15 through 2026-11-21. Its provider IDs match the 658 eligible `FINISHED` result fixtures exactly. The other three result rows belong to two `IN_PLAY` and one `CANCELLED` fixture and are excluded by the feature builder.

### Provider season availability check

Using the configured API key with the existing client, `/competitions/{code}` returned season metadata for all six competitions. The response advertises these counts of distinct season start years (gaps are present; this does not mean every year has a season):

| Competition | Advertised start years | Earliest–latest listed |
|---|---:|---:|
| PL | 128 | 1888–2026 |
| PD | 95 | 1928–2026 |
| SA | 95 | 1929–2026 |
| BL1 | 64 | 1963–2026 |
| FL1 | 83 | 1938–2026 |
| DED | 71 | 1956–2026 |

A direct `/competitions/{code}/matches?season=2025` check succeeded for all six leagues: PL 380 matches, PD 380, SA 380, BL1 306, FL1 306 (305 `FINISHED`, one `AWARDED`), and DED 306 `FINISHED`. The first DED request received HTTP 429; a later retry succeeded, confirming transient rate limiting. Therefore 2025/26 match data is available through the current account for all six competitions. Older season start years are advertised in metadata, but older-season match endpoints were not individually tested. No database imports were performed.

## 2. Feature quality

The CSV contains 49 columns: **39 match-information/pre-match columns** and 10 targets. Missing rates are calculated over all 658 rows. Model usage is based on the saved model feature configurations: Poisson uses six venue-specific rate/count fields; Logistic Regression and XGBoost use 29 numeric and five categorical fields. IDs and match dates are not model inputs.

| Feature | Type | Missing | Poisson | Logistic | XGBoost | Redundancy / quality note |
|---|---|---:|:---:|:---:|:---:|---|
| `provider_match_id` | identifier | 0.0% | — | — | — | Unique row identifier; must remain excluded from predictors. |
| `competition` | categorical | 0.0% | — | Yes | Yes | Only six categories; may carry league-level scoring/home-advantage signal. |
| `season_provider_id` | identifier | 0.0% | — | — | — | Provider identifier; excluded to avoid treating an ID as a numeric quantity. |
| `season_start_year` | numeric | 0.0% | — | Yes | Yes | Only two start years represented; a coarse time proxy and potentially confounded. |
| `match_date` | date | 0.0% | — | — | — | Match date is not passed to the three models; omitting date also omits within-season phase. |
| `home_team_provider_id` | identifier | 0.0% | — | — | — | Stable identifier but excluded; numeric treatment would be meaningless. |
| `home_team` | categorical | 0.0% | — | Yes | Yes | High-cardinality identity category; sparse teams may encourage memorization. |
| `away_team_provider_id` | identifier | 0.0% | — | — | — | Stable identifier but excluded; numeric treatment would be meaningless. |
| `away_team` | categorical | 0.0% | — | Yes | Yes | High-cardinality identity category; sparse teams may encourage memorization. |
| `home_previous_matches_played` | numeric | 0.0% | — | Yes | Yes | Denominator for prior W/D/L and cumulative goal totals; partly redundant. |
| `home_previous_wins` | numeric | 0.0% | — | Yes | Yes | Together with prior draws/losses and matches played, one count is determined by the others. |
| `home_previous_draws` | numeric | 0.0% | — | Yes | Yes | Together with prior wins/losses and matches played, one count is determined by the others. |
| `home_previous_losses` | numeric | 0.0% | — | Yes | Yes | Together with prior wins/draws and matches played, one count is determined by the others. |
| `home_previous_goals_scored` | numeric | 0.0% | — | Yes | Yes | Cumulative volume is team-history length dependent; overlaps with rolling and venue averages. |
| `home_previous_goals_conceded` | numeric | 0.0% | — | Yes | Yes | Cumulative volume is team-history length dependent; overlaps with rolling and venue averages. |
| `home_rolling_avg_goals_scored` | numeric | 9.0% | — | Yes | Yes | Recent five-match view overlaps with cumulative scoring and home scoring rate. |
| `home_rolling_avg_goals_conceded` | numeric | 9.0% | — | Yes | Yes | Recent five-match view overlaps with cumulative conceding and home conceding rate. |
| `home_recent_form` | categorical | 9.0% | — | Yes | Yes | Encodes the same last-five outcomes represented in recent win/draw/loss history; partially redundant. |
| `home_home_matches_played` | numeric | 0.0% | Poisson | Yes | Yes | Denominator for home-only win rate and goal averages. |
| `home_home_win_rate` | numeric | 17.8% | — | Yes | Yes | Related to home-only results and overall prior W/D/L; based on a smaller subset. |
| `home_home_goals_scored_avg` | numeric | 17.8% | Poisson | Yes | Yes | Poisson input; overlaps with all-match and rolling scoring measures. |
| `home_home_goals_conceded_avg` | numeric | 17.8% | Poisson | Yes | Yes | Poisson input; overlaps with all-match and rolling conceding measures. |
| `away_previous_matches_played` | numeric | 0.0% | — | Yes | Yes | Denominator for prior W/D/L and cumulative goal totals; partly redundant. |
| `away_previous_wins` | numeric | 0.0% | — | Yes | Yes | Together with prior draws/losses and matches played, one count is determined by the others. |
| `away_previous_draws` | numeric | 0.0% | — | Yes | Yes | Together with prior wins/losses and matches played, one count is determined by the others. |
| `away_previous_losses` | numeric | 0.0% | — | Yes | Yes | Together with prior wins/draws and matches played, one count is determined by the others. |
| `away_previous_goals_scored` | numeric | 0.0% | — | Yes | Yes | Cumulative volume is team-history length dependent; overlaps with rolling and venue averages. |
| `away_previous_goals_conceded` | numeric | 0.0% | — | Yes | Yes | Cumulative volume is team-history length dependent; overlaps with rolling and venue averages. |
| `away_rolling_avg_goals_scored` | numeric | 8.8% | — | Yes | Yes | Recent five-match view overlaps with cumulative scoring and away scoring rate. |
| `away_rolling_avg_goals_conceded` | numeric | 8.8% | — | Yes | Yes | Recent five-match view overlaps with cumulative conceding and away conceding rate. |
| `away_recent_form` | categorical | 8.8% | — | Yes | Yes | Encodes the same last-five outcomes represented in recent win/draw/loss history; partially redundant. |
| `away_away_matches_played` | numeric | 0.0% | Poisson | Yes | Yes | Denominator for away-only win rate and goal averages. |
| `away_away_win_rate` | numeric | 17.8% | — | Yes | Yes | Related to away-only outcomes and overall prior W/D/L; based on a smaller subset. |
| `away_away_goals_scored_avg` | numeric | 17.8% | Poisson | Yes | Yes | Poisson input; overlaps with all-match and rolling scoring measures. |
| `away_away_goals_conceded_avg` | numeric | 17.8% | Poisson | Yes | Yes | Poisson input; overlaps with all-match and rolling conceding measures. |
| `h2h_previous_matches` | numeric | 100.0% | — | Yes | Yes | All 658 rows are missing; no current sample signal. H2H count/rates also structurally overlap. |
| `h2h_home_team_wins` | numeric | 100.0% | — | Yes | Yes | All 658 rows are missing; no current sample signal; overlaps with H2H draws/losses/count. |
| `h2h_draws` | numeric | 100.0% | — | Yes | Yes | All 658 rows are missing; no current sample signal; overlaps with H2H wins/losses/count. |
| `h2h_home_team_losses` | numeric | 100.0% | — | Yes | Yes | All 658 rows are missing; no current sample signal; overlaps with H2H wins/draws/count. |

**Key findings:** all four H2H fields are 100% missing because the current data never reaches the feature builder's minimum of three prior meetings per pair. They supply no usable variation to the current baselines. Rolling goals and recent-form strings are missing for about 9% of rows; venue rates are missing for 17.8%, reflecting insufficient prior matches rather than fabricated zeros. Logistic Regression/XGBoost impute missing numeric values from training medians and handle missing categories using training-fitted preprocessing. Fully missing H2H numeric columns become constant after preprocessing.

There is structural redundancy: prior wins + draws + losses equal prior matches played; H2H outcomes similarly sum to H2H matches when available. Cumulative goals, rolling five-match averages, and venue averages overlap while describing different windows. Team identity is one-hot encoded for Logistic Regression/XGBoost, but the sample is small relative to the resulting **715 transformed columns**; sparse identity categories risk memorization. `match_date` is excluded, so models receive no explicit within-season phase feature.

## 3. Target distribution

All 658 rows have complete, valid targets. Binary markets show positive/true versus negative/false counts.

| Target | Positive count | Negative count | Positive rate |
|---|---:|---:|---:|
| Home win | 285 | 373 | 43.3% |
| Draw | 155 | 503 | 23.6% |
| Away win | 218 | 440 | 33.1% |
| Over 1.5 | 527 | 131 | 80.1% |
| Over 2.5 | 388 | 270 | 59.0% |
| Over 3.5 | 261 | 397 | 39.7% |
| BTTS | 380 | 278 | 57.8% |

### 1X2 distribution by competition

| Competition | Matches | Home win | Draw | Away win |
|---|---:|---:|---:|---:|
| BL1 | 88 | 43 (48.9%) | 19 (21.6%) | 26 (29.5%) |
| DED | 113 | 37 (32.7%) | 30 (26.5%) | 46 (40.7%) |
| FL1 | 96 | 43 (44.8%) | 22 (22.9%) | 31 (32.3%) |
| PD | 124 | 54 (43.5%) | 32 (25.8%) | 38 (30.6%) |
| PL | 123 | 54 (43.9%) | 33 (26.8%) | 36 (29.3%) |
| SA | 114 | 54 (47.4%) | 19 (16.7%) | 41 (36.0%) |

The small samples differ substantially by competition: home-win rates range from 32.7% in DED to 48.9% in BL1; draw rates range from 16.7% in SA to 26.8% in PL. Combined markets are imbalanced, especially Over 1.5 (80.1%). Baselines and future evaluation should retain probability-quality metrics and class-wise reporting.

## 4. Data integrity

- **Duplicate fixtures:** none found by provider fixture ID; the database also has a unique constraint on that ID.
- **Duplicate results:** none found; result fixture ID is the primary key.
- **Scores:** no negative scores; no score exceeded 15 goals for one side. Observed maximum was 7 home goals, 9 away goals, and 9 total goals.
- **Missing relationships:** no fixture has a missing home team, away team, competition, or season reference. No blank team names or competition codes were found. All seven season records have provider IDs, start dates, and end dates.
- **Same-team fixtures:** none.
- **Team-name collisions:** no normalized duplicate names among the 117 current team records. Historical name changes cannot be checked because the schema stores one current team name, not a name history.
- **Dates:** no kickoff fell more than 45 days before its season start or 90 days after its season end. The dataset date range is 2025-08-15 to 2026-11-21.
- **Non-completed results:** two `IN_PLAY` and one `CANCELLED` fixture have result rows in the raw database. All three are absent from the generated CSV. This is a raw-data status/result inconsistency worth checking against the provider before a future dataset refresh.
- **Target consistency:** all 658 dataset rows' derived 1X2, goal-market, and BTTS targets match stored scores; no target is missing or invalid.

## 5. Feature predictive signal

This is exploratory association only, not a new fitted model. I computed univariate Spearman correlations between numeric pre-match columns and binary outcomes on the **515-row training period only** (dates before 2026-10-31). Target columns were used only as observed outcomes for this diagnostic; they were not included as features. The held-out 143-match period was not used to select or rank features.

Largest absolute correlations were modest:

| Outcome | Strongest numeric association | Spearman rho |
|---|---|---:|
| Home win | Home home-goals-scored average | +0.157 |
| Draw | Away previous losses | +0.075 |
| Away win | Away rolling goals-conceded average | -0.170 |
| Over 1.5 | Away rolling goals-scored average | +0.113 |
| Over 2.5 | Home rolling goals-scored average | +0.116 |
| Over 3.5 | Home home-win rate | -0.097 |
| BTTS | Season start year | +0.123 |

These are weak, unadjusted associations from 515 rows with correlated inputs and multiple comparisons; they do not show causal effects or incremental predictive value. Nonlinear interactions may be missed. Competition outcome spreads above suggest league context may matter, but samples remain small. Overall, this analysis finds some plausible pre-match signal, but not strong standalone relationships in these features.

## 6. Existing model results inspected

Saved evaluations use the same cutoff and 143-match test period:

| Model | Log Loss | Brier | Accuracy |
|---|---:|---:|---:|
| Poisson v1 | 0.9719 | 0.5780 | 51.05% |
| Logistic Regression v1 | 1.1635 | 0.6869 | 41.26% |
| XGBoost v1 | 1.0186 | 0.6079 | 49.65% |

Poisson remains strongest on these aggregate metrics. XGBoost and Logistic Regression use far more features than Poisson's six venue-specific columns, but their current performance does not show that adding columns alone helps. The short, season-skewed dataset and high-dimensional one-hot representation are plausible limitations; this analysis does not establish which caused the performance gap.

## 7. Potential weaknesses and recommended improvements

1. **Historical coverage is the clearest limitation.** Only 658 eligible matches are stored, with just 16 from PL 2025/26; the other six stored season records are 2026/27. After rate limits reset, verify and import earlier provider-advertised seasons, starting with 2025/26 for all six and then older seasons actually served by each endpoint. Record per-season provider counts and reconcile imported rows.
2. **Resolve the three raw status/result anomalies.** Check provider truth for the two in-play and one cancelled fixture before future dataset generation; keep them excluded unless their provider status/result is corrected from source data.
3. **Revisit H2H only after adding history.** At present its four fields are wholly missing. With older seasons ingested, measure H2H coverage before deciding whether it is useful.
4. **Review overlapping feature families using multi-season data.** Test whether cumulative W/D/L counts, match counts, rolling form, and venue rates add distinct value. Do not drop or change features based on this small univariate analysis alone.
5. **Treat team identity carefully.** The 715-column transformed matrix is large relative to 515 training rows; compare identity categories against time-safe team-strength summaries once the sample covers multiple seasons, and check unseen-team behavior.
6. **Expand temporal validation.** The current test period is only 22 days. With additional seasons, use multiple chronological season/period holdouts and calibration/class metrics rather than relying on one short window.
7. **Keep the dataset builder's leakage controls.** It builds rows chronologically and excludes same-UTC-day results from each other's pre-match features. Recheck that invariant after future ingestion or feature changes.

## Scope and files

This report was written under `ml/evaluation/data_quality/`. PostgreSQL data, migrations, the feature pipeline, processed CSV, and all Poisson, Logistic Regression, and XGBoost model/evaluation artifacts were read-only during this analysis. No model was trained or tuned.
