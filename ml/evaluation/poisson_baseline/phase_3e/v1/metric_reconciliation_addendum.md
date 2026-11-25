# Phase 3E.1 — Total-Goals MAE Reconciliation

**Audit scope:** Read-only recalculation from saved rolling-validation predictions. No models were trained, no new predictions were generated, and the locked final test outcomes were not inspected or scored. The original Phase 3E report and artifacts are preserved.

## Finding

The two figures in question come from different prediction runs, even though their validation fixture IDs and fold assignments match.

- `1.3876` for B and `1.3956` for C are the **unweighted means of the four fold MAEs** from the Phase 3E controlled-estimator predictions.
- The saved bootstrap estimate `-0.0160` was calculated from the **Phase 3D benchmark B/C rolling-validation predictions**, not the Phase 3E controlled-estimator predictions. Its sign is C minus B; a negative value means C has lower absolute error.
- Recalculated on Phase 3E controlled-estimator predictions, pooled match-level C minus B MAE is **+0.0028903**, meaning C is slightly worse on this metric. This is not the same comparison as the Phase 3D bootstrap value.

The discrepancy is therefore an estimand/source-run mismatch in the report: it presents the controlled-estimator fold means alongside a paired uncertainty result computed for the earlier Phase 3D B/C estimators. Both use the same 1,602 validation fixture IDs, but different predictions.

## Phase 3E controlled-estimator recalculation

For each match, total-goals absolute error is:

`abs(expected_home_goals + expected_away_goals - actual_total_goals)`

Saved inputs:

- B: `controlled_estimator/B_original_six/folds/fold_{1..4}/predictions.csv`
- C: `controlled_estimator/C_redesigned_13/folds/fold_{1..4}/predictions.csv`

The eight exact file SHA-256 values are listed below. Each B/C pair has identical ordered IDs within its fold. The four folds contain 682, 421, 313 and 186 matches respectively; all 1,602 fixture IDs are unique and align across folds and models.

| Fold | N | B total MAE | C total MAE | C minus B |
|---|---:|---:|---:|---:|
| fold_1 | 682 | 1.2840814 | 1.2795773 | -0.0045041 |
| fold_2 | 421 | 1.2812766 | 1.2874340 | +0.0061574 |
| fold_3 | 313 | 1.4718613 | 1.4675420 | -0.0043194 |
| fold_4 | 186 | 1.5131182 | 1.5478586 | +0.0347403 |

| Aggregation | B | C | C minus B |
|---|---:|---:|---:|
| Mean of four fold MAEs (each fold equal weight) | 1.3875844 | 1.3956030 | +0.0080186 |
| Sample-weighted mean of fold MAEs | 1.3466252 | 1.3495155 | +0.0028903 |
| Pooled match-level MAE | 1.3466252 | 1.3495155 | +0.0028903 |
| Paired mean of match-level C-minus-B absolute-error differences | — | — | +0.0028903 |

The weighted mean equals pooled MAE because folds partition the same set of matches and each match contributes one absolute error. The unweighted fold mean answers a different question by giving the 186-match fold the same weight as the 682-match fold.

## Bootstrap audit

The existing bootstrap inputs are loaded by `_all_validation_predictions(source, "B")` and `_all_validation_predictions(source, "C")` from the Phase 3D source evaluation outputs:

- B: `ml/evaluation/poisson_baseline/expanded_history/v1/folds/fold_{1..4}/predictions.csv`
- C: `ml/evaluation/poisson_baseline/v2/folds/fold_{1..4}/predictions.csv`

Those pairs have identical ordered fixture IDs in each fold and exactly align with the Phase 3E controlled-estimator folds: 682, 421, 313 and 186 rows; 1,602 unique fixtures total. The bootstrap metric is the same absolute error formula. Its per-match vector is second input minus first input, and the caller passes Phase 3D B first and Phase 3D C second, so the actual sign is **Phase 3D C minus Phase 3D B**. It groups by outer-fold × ISO UTC week (31 clusters), resamples paired clusters, and uses 4,000 replicates with seed 73.

Independent recalculation from those Phase 3D files gives:

- B pooled MAE: 1.3655340
- C pooled MAE: 1.3495155
- Paired C-minus-B mean: -0.0160185
- Saved 95% week-block interval: [-0.0471796, +0.0157594]

Thus the saved bootstrap value is arithmetically correct for the Phase 3D outputs. Its generic `B minus A` text in `robustness.json` describes function argument positions, not model names; the report’s “C minus B” label matches the actual caller and subtraction order. The bootstrap interval includes zero and does not establish a reliable difference.

## Source hashes

Phase 3E controlled prediction files:

| File | SHA-256 |
|---|---|
| `controlled_estimator/B_original_six/folds/fold_1/predictions.csv` | `90e16cd8eba93c12a338ae250e7836bd167132d3b40ef6e4d18d17d67c0f454f` |
| `controlled_estimator/B_original_six/folds/fold_2/predictions.csv` | `879c8e34e4e1cb04c9841e205f1fdbadcb7f0b6eb9633ec16e69738485dfd9369` |
| `controlled_estimator/B_original_six/folds/fold_3/predictions.csv` | `6280e660e5b347c6ae301e364019c7ef4b1246921b5ee0000ecc9eb9ca64c38d` |
| `controlled_estimator/B_original_six/folds/fold_4/predictions.csv` | `ad307316c82d88f49ad7db8d0e6841db7ac3a3bda34b8a5e293db84d70b02259` |
| `controlled_estimator/C_redesigned_13/folds/fold_1/predictions.csv` | `82c083e4ec95e60430560b9e2bc9dceabf779c6046e9a91ab5067a82a26a44e9` |
| `controlled_estimator/C_redesigned_13/folds/fold_2/predictions.csv` | `80d5c8ed35d2c9ea044c41fb561c7075a59a5b14ea6bafc642c5b31b8a7933d0` |
| `controlled_estimator/C_redesigned_13/folds/fold_3/predictions.csv` | `76376635effa81520d6822abfb31b2d479d373f8a04fef1b5d4adc41d2cea8e7` |
| `controlled_estimator/C_redesigned_13/folds/fold_4/predictions.csv` | `208bb6f51ef58237b02a7bac2499579b572771f03f8414e732a2b40082d83595` |

The eight Phase 3D source prediction hashes are preserved in the existing `configuration.json` `source_output_hashes` map and were verified against the saved files. The Phase 3D B/C model, evaluation, and prediction hashes also remain covered by that source map and the Phase 3D run manifests.

Before adding this correction, deterministic directory fingerprints (SHA-256 over sorted relative-path and per-file SHA-256 lines) were:

| Artifact tree | Files | Fingerprint |
|---|---:|---|
| Phase 3E evaluation `ml/evaluation/poisson_baseline/phase_3e/v1` | 214 | `a19890e137afbdb25e6da591fb56927ca6c4dbb02176c6c48133dc011256baff` |
| Phase 3E diagnostics `ml/models/poisson_baseline/phase_3e/v1` | 80 | `3db795800dc4ae637fe31dc478191feb6796d4c076c562c44ed6f05961dfdbfa` |
| Phase 3D B evaluation `ml/evaluation/poisson_baseline/expanded_history/v1` | 23 | `8357889dd104f874b933968677d8c95e450c6980ad8daf5ce870bfc25ba69109` |
| Phase 3D C evaluation `ml/evaluation/poisson_baseline/v2` | 24 | `008133b7a8b6cb455e42eccc6aa86288a6f2b35ed607ddc2b77ec7bdcefc5dec` |
| Phase 3D B model `ml/models/poisson_baseline/expanded_history/v1` | 3 | `09a636dddfccf221d2e8b930aedb66df6bb01dde453c0b90525154a5abba158f` |
| Phase 3D C model `ml/models/poisson_baseline/v2` | 6 | `d292ee74372db39e3d987a9d2e6db8e0797076b57279296386c4fe070908b83f` |

These tree fingerprints exclude this addendum; all pre-existing files remain unchanged. The original Phase 3E report is preserved.

## Conclusion

There is no arithmetic error in either calculation. The equal-fold controlled comparison slightly favors B on pooled total-goals MAE (+0.0029 for C minus B), while the Phase 3D model comparison favors C by 0.0160. The original report combined values from distinct prediction runs without making that distinction clear. Treat both differences as uncertain; the saved Phase 3D bootstrap interval crosses zero.
