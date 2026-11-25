# Phase 3G.2.2 — Linux CI Verification

## Outcome: blocked; security gate remains open

The hosted workflow was triggered after publishing the project. It did **not** pass, and the available GitHub summary does not expose enough output to determine the pytest root cause or whether the individual symlink tests ran. No registry or workflow code was changed in response.

| Check | Result |
|---|---|
| Repository/reference | `mntez/frprediction`, branch `codex/publish-frprediction-project` |
| Commit tested | `af576671ac7cb6b52b5fe8e30c20243a49abcf97` |
| GitHub Actions run | [45554416162](https://github.com/mntez/frprediction/actions/runs/45554416162) |
| Ubuntu job | Failed at `Run model registry tests`; process exit code 4 |
| Windows hosted job | Failed at `Run model registry tests`; process exit code 1 |
| Hosted test counts | Unavailable from the run summary |
| Hosted file-symlink test | Exact result unavailable; cannot confirm pass or skip |
| Hosted directory-symlink test | Exact result unavailable; cannot confirm pass or skip |
| Local Windows focused suite | 14 passed, 2 skipped |
| Local symlink cases | `test_symlink_outside_model_root_is_rejected` and `test_symlinked_directory_outside_model_root_is_rejected` both skipped because local Windows denied symlink creation |
| Local full backend suite | 88 passed, 2 skipped |

## Diagnosis available

GitHub's job metadata confirms setup, checkout, Python setup, and dependency installation all completed successfully on both operating systems. The pytest command step then exited nonzero. The public run page shows exit codes but requires sign-in to view logs. The GitHub Actions job-log API returned HTTP 403 with `Must have admin rights to Repository`.

Therefore the observed failure is in the CI test command, but its underlying cause cannot be diagnosed from the accessible output. In particular, exit code 4 alone does not establish whether Ubuntu collected or executed either symlink test. No test count or symlink result is inferred.

## Workflow configuration

`.github/workflows/model-registry-security.yml` is present on the tested branch. It runs the registry test file on an OS matrix containing `ubuntu-latest` and `windows-latest`, using Python 3.12. The local Windows results demonstrate that the ordinary registry tests remain compatible; Windows skipped only the two OS-dependent symlink creation cases locally.

## Next verification needed

An account with access to the run logs must provide the failed pytest output (or grant the required Actions log access). After diagnosing the output, any proposed fix should be reviewed separately, as requested. Then rerun the Ubuntu job and verify both named symlink tests are reported as passed, not skipped. Keep the registry research-only and do not start Phase 3G.3 until that verification succeeds.

No model training, prediction generation, activation, database change, or protected-artifact modification was performed for this verification update.
