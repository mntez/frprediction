# FRPRECTIONS Phase 3B — Independent v2 Feature and Dataset Audit

**Audit date:** 2026-11-24  
**Scope:** Read-only inspection and independent recalculation. No source code, dataset, PostgreSQL record, model artifact, or frontend file was changed. No model was trained and no pytest suite was run.

## Executive verdict

| Area | Finding |
|---|---|
| v2 dataset integrity, target correctness, and hashes | **PASS** |
| v2 feature formulas and point-in-time state | **PASS**; independent replay matched all 13 feature values for all 2,699 rows |
| Awarded fixture exclusion from the v2 dataset/state | **PASS** |
| Strict kickoff-time ordering within a UTC date | **WARNING**; rows use UTC date then provider ID, not kickoff time within that day |
| Comparator B input readiness | **FAIL**; the available expanded CSV carries 542704 into six later legacy input rows, and no safe B-specific trainer exists |
| Train-fold preprocessing readiness | **WARNING**; no v2/B training preprocessing is implemented in this phase, so train-only fitting is not yet enforceable |
| **Overall model-training readiness** | **FAIL** until comparator B data and isolated training paths are corrected/implemented |

The v2 CSV itself is internally consistent and reproduces the approved formulas. The main blocker is comparator B: reusing the current legacy CSV would violate the awarded-match exclusion, and reusing the existing Poisson training script would write into protected v1 artifact paths.

## 1. Dataset integrity

Independent checks against the saved CSV, manifest, v1 test-prediction file, and read-only PostgreSQL queries found:

| Check | Result |
|---|---|
| Rows / columns | 2,699 / 32 |
| Match-information / feature / target columns | 9 / 13 / 10 |
| Training / test rows at cutoff 2026-10-31 | 2,556 / 143 |
| Unique fixture IDs / duplicate full rows | 2,699 / 0 duplicates |
| Fixture IDs equal all database `FINISHED` fixture IDs | Yes |
| Test IDs equal saved Poisson v1 ordered test IDs | Yes, exact order |
| Comparator B and v2 ordered train/test manifests | Identical in the manifest |
| Fixture 542704 in v2 CSV | No |
| All target values recomputed from database scores | 0 mismatches |

Database result-bearing statuses were 2,699 `FINISHED`, one `AWARDED`, two `IN_PLAY`, and one `CANCELLED`. The v2 output contains only the 2,699 `FINISHED` fixtures.

### Hash verification

Hashes were recalculated from the bytes/records rather than accepted from the manifest:

| Hash | Recalculated value | Manifest/reported value |
|---|---|---|
| Source-data SHA-256 | `ddb63a3885c435049f51a7acfcc4a820bcd883e91acfd38f7ee1bedf7d0124bb` | Match |
| Dataset SHA-256 | `7656a4694f315f02913e55260d7599f7034ab27b5a671ca10ffd902375a461d5` | Match |
| Configuration SHA-256 | `976ea5e8a12c84e4b001f8481f87c4cca021bfc2db3478c3115fe661055e41d8` | Match |
| Manifest file SHA-256 | `67daae2ea0361ccdc8ede1f1630b9cf0e518068dee0103f9fc040733b59317e6` | Recalculated for this audit; not self-stored |

The ordered fixture-ID lists and comparator split lists are in `ml/data/processed/v2/manifest.json`.

### Chronological ordering

Rows are monotonically ordered by UTC `match_date`, with provider fixture ID as the tie-breaker within each date. The generator first sorts by kickoff, groups all fixtures by UTC date for feature calculation, then re-sorts saved rows by `(match_date, provider_match_id)`. Therefore feature chronology and the UTC-date leakage rule pass, but the saved row order is not strict kickoff-instant order within a date: the audit found 911 adjacent same-day inversions across 220 multi-fixture date groups. This does not affect generated features because all same-date fixtures use the same opening-of-day state. **Warning:** document this ordering contract or preserve kickoff order if “chronological order” is intended to mean exact kickoff timestamps.

## 2. Feature-by-feature formula audit

I independently replayed the eligible database matches in UTC-date batches and recalculated every v2 feature for every CSV row. Result: **2,699 rows × 13 features checked, zero mismatches**.

| Feature(s) | Observed calculation | Finding |
|---|---|---|
| `home_attack_strength`, `away_attack_strength` | For each prior same-competition appearance, normalize goals-for by the then-available venue scoring mean. Apply 365-day half-life weights and neutral prior 1 with 5 pseudo-observations: `(5 + Σ(w·ratio))/(5 + Σw)`. | **PASS** |
| `home_defense_strength`, `away_defense_strength` | Normalize goals-against by the opponent’s venue mean, then invert the shrunk concession ratio: `(5 + Σw)/(5 + Σ(w·concession_ratio))`. | **PASS** |
| `home_form_points_5`, `away_form_points_5` | Mean of 3/1/0 points over the latest up-to-five prior same-competition appearances, all venues. | **PASS** |
| `home_goals_for_5`, `away_goals_for_5` | Mean goals scored over the same up-to-five prior appearances. | **PASS** |
| `home_goals_against_5`, `away_goals_against_5` | Mean goals conceded over the same up-to-five prior appearances. | **PASS** |
| `elo_difference` | Home opening-of-date rating minus away opening-of-date rating, in separate competition pools. New team prior is 1500; update uses scale 400, `K=10`, home advantage 60, and draw score 0.5. | **PASS** |
| `home_history_matches`, `away_history_matches` | Count of prior same-competition `FINISHED` appearances across venues. | **PASS** |

Competition baselines are expanding means of prior same-competition home goals and away goals, separately. Each requires at least 20 prior fixtures in that role; zero mean is also treated as unavailable to avoid division by zero. Baselines and histories update only after a UTC-date batch. Historical goal ratios use the baseline available before the historical appearance; later league means are not backfilled. A team with no usable normalized history receives neutral strength 1 only after the competition baseline is available; otherwise strength is missing.

Elo season regression is applied when a team first appears in a later season: `1500 + (rating - 1500) * 0.75**season_year_gap`. Competition and team IDs key the rating pools, so a team changing competition starts at that competition’s 1500 prior. Same-date rating changes are calculated from `day_ratings` snapshots, aggregated in `elo_deltas`, and applied after all date features are emitted.

Venue residuals are not among the 13 saved features. The optional helper is present in the module and filters by venue with a five-observation minimum, but is not used by the v2 dataset builder.

## 3. Missingness investigation

Independent CSV parsing confirmed:

| Feature group | Missing values |
|---|---:|
| Each of the four team-strength features | 128 / 2,699 = 4.74% |
| Each home form/goals feature | 67 / 2,699 = 2.48% |
| Each away form/goals feature | 64 / 2,699 = 2.37% |
| Elo difference and both exposure counts | 0% |
| All ten target columns | 0% |

All four strength features have the same 128 missing fixture IDs. Every one has fewer than 20 prior same-competition fixtures, so the competition baseline is unavailable. There are **no unexpected strength nulls after the baseline is available**. In 71 of those rows both teams already have prior-match exposure, which confirms baseline warm-up—not lack of team history—is the operative reason. The other 57 rows have no prior match for either team as well as an unavailable baseline. Form/goals nulls occur only when the corresponding team has zero prior appearances.

All strength missingness is concentrated in season start year 2025; none occurs in the 2026 season rows.

| Competition | Missing fixture dates | Fixture IDs with missing strength |
|---|---|---|
| BL1 | 2025-08-22–2025-09-13 (25) | `540406, 540407, 540408, 540409, 540412, 540413, 540414, 540410, 540411, 540423, 540416, 540417, 540418, 540420, 540421, 540415, 540419, 540422, 540425, 540424, 540426, 540427, 540429, 540430, 540432` |
| DED | 2025-08-08–2025-08-23 (21) | `537441, 537442, 537443, 537444, 537445, 537446, 537447, 537448, 537449, 537450, 537451, 537452, 537453, 537454, 537455, 537456, 537457, 537458, 537459, 537460, 537461` |
| FL1 | 2025-08-15–2025-08-30 (22) | `542414, 542407, 542410, 542411, 542406, 542408, 542409, 542412, 542413, 542422, 542418, 542419, 542420, 542415, 542416, 542417, 542421, 542423, 542431, 542426, 542427, 542429` |
| PD | 2025-08-15–2025-08-25 (20) | `544214, 544220, 544211, 544216, 544219, 544212, 544213, 544215, 544217, 544218, 544225, 544222, 544226, 544227, 544223, 544224, 544228, 544230, 544221, 544229` |
| PL | 2025-08-15–2025-08-25 (20) | `537785, 537786, 537787, 537788, 537789, 537790, 537791, 537792, 537793, 537794, 537804, 537795, 537797, 537798, 537799, 537802, 537796, 537800, 537801, 537803` |
| SA | 2025-08-23–2025-08-31 (20) | `536815, 536816, 536819, 536823, 536817, 536818, 536820, 536822, 536821, 536824, 536833, 536834, 536826, 536830, 536831, 536832, 536825, 536827, 536828, 536829` |

Missing values are preserved as empty CSV cells; no fabricated score or team statistic is substituted. The manifest’s missing-rate fields agree with independently parsed CSV values.

## 4. Leakage audit

| Safeguard | Finding |
|---|---|
| Same-UTC-date batch | **PASS.** Matches are sorted by UTC kickoff, grouped by UTC date, features are emitted for the whole date, and only then are state updates applied. |
| Own-result/future-result exclusion | **PASS.** Feature rows read preexisting histories/baselines; target labels are created separately. State mutation occurs after the entire date batch. Independent feature replay matched every saved value. |
| Fixture 542704 in v2 history/targets | **PASS.** `build_training_rows_v2` accepts only status `FINISHED`; 542704 is `AWARDED`, absent from the v2 IDs and targets. |
| Elo pre/post separation | **PASS.** Pre-date ratings are saved in `day_ratings`; deltas are accumulated separately and applied after features. The dataset retains only their pre-match difference. |
| Later validation-period state | **PASS for feature generation.** A full chronological dataset lets earlier eligible validation results feed later validation dates, without same-date updates. There is no separate v2 rolling-origin validation runner yet. |
| Training-only preprocessing | **WARNING / not yet implemented.** This dataset builder performs no imputation, scaling, or encoding. The manifest documents a future fold-only policy, but no B/C preprocessing code currently enforces it. |

The original database record for fixture 542704 remains `AWARDED`, 0–0, winner `DRAW`; this audit issued only SELECT queries. No write or regeneration command was run.

## 5. Target and eligibility audit

Every dataset ID maps to a current database fixture with status `FINISHED`; each of the ten targets was recalculated from its stored home/away score, with zero mismatches. The only AWARDED, two IN_PLAY, and one CANCELLED result-bearing rows are excluded. The test fixture list matches the saved Poisson v1 test list exactly and is unchanged. No PostgreSQL mutation was performed.

## 6. Comparator B readiness

**Feature definition check — PASS:** `ml/baselines/poisson.py` still defines the original six `TEAM_FEATURES`, and `PoissonBaseline` consumes those legacy columns explicitly. It does not need v2-only features.

**Actual B data/trainer readiness — FAIL:** the available `ml/data/processed/training_dataset.csv` was generated by the legacy builder, whose `COMPLETED_STATUSES` includes both `FINISHED` and `AWARDED`. Filtering that CSV to the 2,556 manifest IDs removes the awarded row as a target but does not remove its effect from later historical feature values.

I rebuilt the original six legacy inputs in memory from `FINISHED` records only and compared them against the expanded CSV. **18 feature values across six later fixtures differ** (three away-venue inputs each); all are Toulouse FC away fixtures and the differences trace to fixture 542704:

| Fixture | Date | Legacy away appearances | FINISHED-only appearances | Split |
|---:|---|---:|---:|---|
| 559710 | 2026-08-29 | 17 | 16 | Train |
| 559684 | 2026-09-12 | 18 | 17 | Train |
| 559673 | 2026-10-10 | 19 | 18 | Train |
| 559661 | 2026-10-16 | 20 | 19 | Train |
| 559645 | 2026-11-01 | 21 | 20 | Test |
| 559623 | 2026-11-21 | 22 | 21 | Test |

The legacy average goals-for/goals-against values also differ for these rows. Thus the manifest proves B/C fixture-ID equality, but does not prove B feature-state eligibility.

There is no dedicated B trainer. The existing `ml/train_poisson_baseline.py` reads the unfiltered legacy CSV, splits by date, and writes to `ml/models/poisson_baseline/v1/` and `ml/evaluation/poisson_baseline/v1/`. Reusing it for B would train on the awarded target and risk overwriting protected v1 artifacts. Do not use that script for B.

## 7. Differences, risks, and recommended corrections

1. **Comparator B must be isolated before training.** Add a B-specific input build that recomputes the original six features from `FINISHED`-only history, filters targets to the same 2,556 IDs, and asserts the same 143 test IDs. Do not merely filter the legacy CSV by ID.
2. **Give B separate artifact destinations.** A dedicated script must write only under the approved `expanded_history/v1/` directories and assert the exact six-column allowlist. Preserve the existing Poisson v1 trainer and output paths.
3. **Implement and test preprocessing isolation when training starts.** Fit all imputation/scaling/encoding parameters on each training fold; do not treat manifest prose as enforcement.
4. **Clarify same-day output ordering.** Current CSV order is UTC date then provider ID, while state is correctly same-date batched. If strict kickoff order is required, record/emit it without allowing fixtures on the same date to update each other.
5. **Minor safe extension to design:** a competition mean of zero is treated as unavailable, in addition to the approved 20-observation minimum. This prevents division by zero and caused no unexpected missing rows in the current data.

## 8. Files inspected

- `AGENTS.md`
- `ml/data/processed/v2/training_dataset.csv`
- `ml/data/processed/v2/manifest.json`
- `ml/features/feature_engineering_v2.py`
- `backend/app/jobs/build_training_dataset_v2.py`
- `backend/tests/test_feature_engineering_v2.py`
- `ml/evaluation/feature_architecture_redesign/design_report.md`
- `ml/data/processed/training_dataset.csv`
- `ml/features/feature_engineering.py`
- `ml/baselines/poisson.py`
- `ml/train_poisson_baseline.py`
- `ml/evaluation/poisson_baseline/v1/test_predictions.csv`
- Relevant SQLAlchemy fixture/result/season/team models and current database rows, queried read-only.

## Final readiness verdict

- **v2 feature dataset:** **PASS** for the 2,699-row, 2,556/143 split; independent feature recalculation had zero mismatches, targets and hashes match, and 542704 contributes to no v2 feature or target.
- **Comparator B:** **FAIL** until its legacy six inputs are rebuilt from eligible `FINISHED` history and a separate trainer/output path is provided.
- **Model training:** **NOT READY** for the planned comparison phase. Keep model training paused until B is corrected and fold-only preprocessing is implemented and verified. No correction was applied in this audit.
