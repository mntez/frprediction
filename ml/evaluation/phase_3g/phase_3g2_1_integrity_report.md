# Phase 3G.2.1 — Registry Integrity and Security Closure

## Result

Registry integrity checks pass against the existing recorded baselines. The previous Phase 3G.2 report contained a mistyped Phase 3E evaluation fingerprint; the valid value is independently recorded in the Phase 3G.1.1 clarification report and matches the current tree. Secure file and directory symlink tests are now included, with Linux CI configured to exercise both.

No model, evaluation, dataset, database, frontend, or dependency artifacts were changed. The only executable changes are the symlink test and its CI workflow.

## Phase 3E fingerprint investigation

- The repository has no `.git` directory, so Git history could not be queried.
- A search found no backup files (`.bak`, `.backup`, `.old`, `.orig`, `.patch`, or `.zip`) containing an alternative snapshot.
- The existing `ml/evaluation/phase_3g/architecture_clarification_report.md` records the Phase 3E evaluation tree as 215 files with fingerprint `3d82f74c080af630dee89afd640f315d9c27093f2c7d10e64d382a23ffdd6bde`. Recomputing the fingerprint over sorted relative-path/SHA-256 lines produces the same value.
- `ml/evaluation/phase_3g/phase_3g2_implementation_report.md` previously contained a malformed transcription (`3d82f74c080af630dee89afd640f315d9c2707c10e64d382a23ffdd6bde`). It has been corrected.
- The Phase 3E.1 metric addendum records a distinct 214-file snapshot (`a19890e137afbdb25e6da591fb56927ca6c4dbb02176c6c48133dc011256baff`) from before that addendum was added. It is not the baseline for the current 215-file directory.

The valid Phase 3G.1.1 fingerprint is therefore the authoritative recorded baseline; historical equality is established. The current 215-file tree is also the new verified baseline for future checks.

## Symlink containment

The loader resolves candidate paths and verifies that each resolved file remains under the resolved configured model root. No path-validation code was weakened.

The registry tests now cover:

- A file symlink inside the model root pointing to a file outside it.
- A directory symlink inside the model root pointing to a directory outside it.
- Existing traversal and missing-path cases.

Both symlink cases are run on `ubuntu-latest` in `.github/workflows/model-registry-security.yml`. The workflow also runs the focused registry suite on `windows-latest`; Windows skips a symlink case only when the runner denies symlink creation. The workflow uses Python 3.12 and installs only pytest, joblib, NumPy, SciPy and the supported scikit-learn range.

Local Windows result: `py -m pytest tests/test_model_registry.py -q` from `backend` — **14 passed, 2 skipped**. Both skips are expected because this Windows environment does not permit symlink creation. Linux-specific behavior is configured in CI but has not yet executed in this local environment.

## Integrity verification

Tree fingerprints use SHA-256 over sorted lines of `relative/path file_sha256\n`. File hashes are SHA-256 of exact file bytes.

| Protected tree | Files | Recorded baseline | Current fingerprint | Result |
|---|---:|---|---|---|
| Poisson A model `ml/models/poisson_baseline/v1` | 1 | `7aef45397fc004889741fedf13797e8d7e48fddf95916faa088f1a3363dbf2a3` | same | PASS |
| Poisson A evaluation `ml/evaluation/poisson_baseline/v1` | 2 | `a36ce4eab8fd2ea4fbe36990472c457f59a7dd377aa02717f3e3b7ad1afccc85` | same | PASS |
| Poisson B model `ml/models/poisson_baseline/expanded_history/v1` | 3 | `09a636dddfccf221d2e8b930aedb66df6bb01dde453c0b90525154a5abba158f` | same | PASS |
| Poisson B evaluation `ml/evaluation/poisson_baseline/expanded_history/v1` | 23 | `8357889dd104f874b933968677d8c95e450c6980ad8daf5ce870bfc25ba69109` | same | PASS |
| Poisson C model `ml/models/poisson_baseline/v2` | 6 | `d292ee74372db39e3d987a9d2e6db8e0797076b57279296386c4fe070908b83f` | same | PASS |
| Poisson C evaluation `ml/evaluation/poisson_baseline/v2` | 24 | `008133b7a8b6cb455e42eccc6aa86288a6f2b35ed607ddc2b77ec7bdcefc5dec` | same | PASS |
| Comparator B dataset `ml/data/processed/comparator_b/v1` | 2 | `c5b2f9f6eb7076fa07949c29e5f23828f207e966fa666b0f55e66249d203739c` | same | PASS |
| v2 dataset `ml/data/processed/v2` | 2 | `20b8a443023b6a30b650641d68f205ba7f188b2b6c7dfa4b7a3b8f356e732451` | same | PASS |
| Phase 3E evaluation `ml/evaluation/poisson_baseline/phase_3e/v1` | 215 | `3d82f74c080af630dee89afd640f315d9c27093f2c7d10e64d382a23ffdd6bde` | same | PASS |
| Phase 3E model diagnostics `ml/models/poisson_baseline/phase_3e/v1` | 80 | `3db795800dc4ae637fe31dc478191feb6796d4c076c562c44ed6f05961dfdbfa` | same | PASS |

### Registry metadata and configuration

The registry remains explicit and research-only; no candidate was approved or activated. Its current source-file SHA-256 is `a5533d05091a1e4241bf538508ab34dc400a339edb7073a81d26fd13a0a8b1f5`.

The model-specific file hashes were also recomputed and match the hashes pinned in the registry:

| Model/artifact | SHA-256 |
|---|---|
| A `model.json` | `77b138a6014601b19ee32021da3ce80039cafe82270d8d890b524636d3facd1a` |
| B `model.json` | `defc0757c8282d6008c68bcfcc0a73ab4c1f7f3b76be37e97bd462c94dd85861` |
| C `home_goals_model.joblib` | `435dc6a174dfefbeaf6f31c54c5648beff6fe2598b62acd28a0dd0b9721892d7` |
| C `away_goals_model.joblib` | `aada530c957fac5f38372f78bcf59d353d8664e5b122d3a665448032a878c415` |
| C `preprocessor.joblib` | `eedf6fd40fe5a7c8dc19978979072a71bf567ff9da7c0609a2c7802b0f69ea99` |
| C `model_metadata.json` | `daf56c22f6f6cb1cf6055ced1c27c7402c3f987205717746456f71a05e8c7a1f` |
| C `training_config.json` | `b2e4087105ad8b077dcf5bcfe74ecd476c3570f6d24afea973c342f3c3dfcb08` |
| C `fixture_manifest.json` | `a076edef3f61024262e0554d908d0deec9336831e83c9d8177088413dd95204a` |

Independent Phase 3E run metadata remains consistent: `manifest.json` records configuration SHA-256 `d8ee5716aa08dc3d9b0d04f993e947349b25f283863324f24cad5f9b127a1e1a`, B dataset SHA-256 `b33d54eb7da5eed619d97b054d1ca29031f8533bf3bf2b9729f1a4dc7661d09e`, and C dataset SHA-256 `7656a4694f315f02913e55260d7599f7034ab27b5a671ca10ffd902375a461d5`. The registry source hash is reported as current integrity metadata, not a historical pre-Phase-3G baseline.

## Tests and commands

- `py -m pytest tests/test_model_registry.py -q` from `backend`: **14 passed, 2 skipped** on Windows.
- `py -m pytest -q` from `backend`: **88 passed, 2 skipped** on Windows; both skips are symlink creation cases.
- Linux execution is pending the added GitHub Actions run. The repository currently contains no `.github` history or existing CI workflow.

No models were trained, no predictions were generated, no model was activated, and no database or migration was changed.

## Files changed or created

- Updated `backend/tests/test_model_registry.py` with directory-symlink containment coverage.
- Added `.github/workflows/model-registry-security.yml` for Ubuntu and Windows registry test runs.
- Created this report.
- Corrected the Phase 3G.2 report's Phase 3E fingerprint explanation and value.

## Remaining limitations

- Git-based recovery was unavailable because this working directory has no `.git` metadata.
- The Linux CI configuration is present, but its first hosted run has not yet completed.
- Windows skipped both symlink creation tests due to local OS privileges; Linux CI is the environment intended to execute those paths.
