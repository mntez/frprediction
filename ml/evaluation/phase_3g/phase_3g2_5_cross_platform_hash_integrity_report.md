# Phase 3G.2.5 — Cross-Platform Registry Hash Integrity

## Outcome

Cross-platform hash integrity is fixed without changing the registry's expected hashes or weakening verification. Git attributes now preserve exact bytes for the eight text files whose hashes the registry pins. The latest hosted workflow passed on both platforms.

Successful run: [45590598690](https://github.com/mntez/frprediction/actions/runs/45590598690)  
Branch: `codex/publish-frprediction-project`  
Commit: `33e1d724cecb7aa34da60ebdb49f80b2bba1a786`

## Previous failures and exact inputs

The previous run was [45586717471](https://github.com/mntez/frprediction/actions/runs/45586717471), commit `0ffda30b31fcf3dcf964c491e8eb0e66ab74b239`. Both jobs collected 16 tests, with 4 failures and 12 passes. The same four test cases failed on each OS:

| Test | Ubuntu mismatch | Windows mismatch |
|---|---|---|
| `test_registered_a_loads_original_multiplicative_contract_from_temp_artifacts` | `ml/models/poisson_baseline/v1/model.json`: expected `77b138a6014601b19ee32021da3ce80039cafe82270d8d890b524636d3facd1a`; got `c4185b68c1364b60364d4d8f092228540b393aa1cdfb79d442f0a75d34c87343` | `ml/baselines/poisson.py`: expected `076014cfdcd9287338ed5d8a28d8a03cb1e568392cc3db12375a77e2e6133fa7`; got `5965e6271de8e518e3d25655e9f664588a407d3cf4eb5424685afecf733a186d` |
| `test_registered_c_loads_both_fitted_models_and_preprocessor_without_fitting_or_predicting` | `ml/models/poisson_baseline/v2/model_metadata.json`: expected `daf56c22f6f6cb1cf6055ced1c27c7402c3f987205717746456f71a05e8c7a1f`; got `190e6679138ba01d01a1e8733444e965be768c188129339a62c1cfd209830e59` | `ml/features/feature_engineering_v2.py`: expected `55ac1b291965a31b39fb6a58142e5450c83d3d854a037413aa25c3e87631da7a`; got `1b8eb8678f989fb520a48393ca26c82fec7b2df0778a333c6ac9909c45d6eb3b` |
| `test_c_feature_configuration_mismatch_fails_after_integrity_check` | Same C `model_metadata.json` mismatch | Same C feature-generation source mismatch |
| `test_loading_does_not_modify_existing_a_b_c_or_phase3e_artifacts` | Same A `model.json` mismatch | Same A formula-source mismatch |

The Ubuntu runner's four failures were against the two JSON artifacts; the Windows runner's four were against the two Python source files. The other C JSON artifacts were also at risk: `training_config.json` and `fixture_manifest.json` were LF-normalized in the prior Git blobs, though loading stopped at `model_metadata.json` before reporting their mismatches.

## Root cause and evidence

The registry hashes exact file bytes. There was no `.gitattributes` or `.editorconfig`; the workflow did not set checkout line-ending options. The development Git configuration had `core.autocrlf=true`.

- The original A and C JSON files contain CRLF. Git's prior text normalization stored LF bytes in the commit, so Ubuntu read the LF Git blobs and calculated the `c418…` and `190e…` hashes.
- The hash-pinned Python source files contain LF. Windows checked them out as CRLF, producing the `5965…` and `1b8e…` hashes.
- The mismatch occurred at file-byte hashing in the artifact loader. It was not an inference, deserialization, or test invocation problem.

For the four JSON artifacts, committed blob hashes before the fix and after the fix are:

| File | Before (LF-normalized blob) | After (exact original bytes) | Registry expected |
|---|---|---|---|
| A `ml/models/poisson_baseline/v1/model.json` | `c4185b68c1364b60364d4d8f092228540b393aa1cdfb79d442f0a75d34c87343` | `77b138a6014601b19ee32021da3ce80039cafe82270d8d890b524636d3facd1a` | same as after |
| C `ml/models/poisson_baseline/v2/model_metadata.json` | `190e6679138ba01d01a1e8733444e965be768c188129339a62c1cfd209830e59` | `daf56c22f6f6cb1cf6055ced1c27c7402c3f987205717746456f71a05e8c7a1f` | same as after |
| C `ml/models/poisson_baseline/v2/training_config.json` | `049aebad715ca6f3094c81cda38e94c3de5aca90583c70439e606488f17ea39c` | `b2e4087105ad8b077dcf5bcfe74ecd476c3570f6d24afea973c342f3c3dfcb08` | same as after |
| C `ml/models/poisson_baseline/v2/fixture_manifest.json` | `ea1e56baba9f46ad6a0fe9556d25e0ae930512e3b470dd27909e301499cda091` | `a076edef3f61024262e0554d908d0deec9336831e83c9d8177088413dd95204a` | same as after |

The Python source blobs already matched their pinned LF hashes before the fix. The Windows working-tree variants for the two failing files matched CRLF conversion exactly. After the fix, committed and working-tree hashes of all eight pinned text files match the existing registry values. No expected hash or hash manifest was regenerated.

## Correction

Added narrowly scoped rules in `.gitattributes`:

- Mark the four hash-pinned JSON artifact paths `-text`, preserving their existing CRLF bytes exactly in Git and on every OS.
- Mark the four hash-pinned Python source paths `text eol=lf`, preserving LF checkouts on Windows.

No broad JSON or Python normalization rule was added. No registry code or tests changed. The four JSON working-tree files were byte-identical to the original source files before and after; Git now stores those same original bytes instead of the normalized LF variants. Model values, datasets, and evaluation contents did not change. The registry still performs exact SHA-256 checks.

## Verification

The corrected workflow installed dependencies from `backend/requirements.txt`, ran `python -m pytest tests/test_model_registry.py -q`, and completed successfully:

| OS | Result | Skips |
|---|---:|---:|
| Ubuntu | **16 passed** | 0 |
| Windows | **16 passed** | 0 |

The suite defines 16 tests. The log's quiet-mode summary reports all 16 passed with no skips, so both symlink test cases executed and passed on each runner:

- `test_symlink_outside_model_root_is_rejected` — passed on Ubuntu and Windows.
- `test_symlinked_directory_outside_model_root_is_rejected` — passed on Ubuntu and Windows.

Local Windows verification after the attributes change: `py -m pytest tests/test_model_registry.py -q` from `backend` — **14 passed, 2 skipped**. Those local skips remain due to Windows symlink-creation privileges; hosted Windows and Ubuntu both executed those cases successfully.

## Files changed and limitations

- Added `.gitattributes` with eight path-specific rules.
- The four JSON Git blobs listed above were restored to their original, registry-pinned bytes; their values and local file contents were not changed.
- Created this report.

No model training, prediction generation, activation, database change, migration, registry implementation change, or dataset/evaluation artifact change occurred. No hash-baseline or manifest update was needed. No unresolved cross-platform hash issue remains for the registered A/C text artifacts and source files.
