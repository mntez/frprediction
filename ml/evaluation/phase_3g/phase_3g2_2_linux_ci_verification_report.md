# Phase 3G.2.2 — Linux CI Verification

## Outcome: Ubuntu run could not be dispatched

The workflow was **not run on Ubuntu**, so there is no workflow run URL, Git reference, or Ubuntu test count to report. The workspace has no `.git` directory or configured remote, and GitHub CLI and Docker are unavailable. WSL is installed as a Windows command but no Linux subsystem is installed. Without a GitHub repository reference or local Linux runner, this workspace cannot start the hosted Ubuntu job.

This is an environment limitation, not a reported workflow failure. No code changes were made in response.

## Workflow and Windows verification

Inspected `.github/workflows/model-registry-security.yml`. It triggers on push and pull request and has an OS matrix of `ubuntu-latest` and `windows-latest`. Both jobs use Python 3.12 and run the focused registry suite from `backend`:

`python -m pytest tests/test_model_registry.py -q`

The Windows job remains configured. Running the same command locally on Windows produced:

- **14 passed, 2 skipped**.
- `test_symlink_outside_model_root_is_rejected` — skipped because Windows denied symlink creation.
- `test_symlinked_directory_outside_model_root_is_rejected` — skipped because Windows denied symlink creation.

Therefore, Windows compatibility for the ordinary registry tests is confirmed; neither symlink containment case actually executed locally. Ubuntu execution is specifically needed to verify those two tests run rather than skip.

## Required to complete hosted verification

The workflow file must be present in a GitHub-hosted repository and triggered by a push or pull request. After that run completes, record its run URL and commit SHA, confirm both named symlink tests show as passed (not skipped), and capture the full registry test counts. No model, dataset, evaluation artifact, registry metadata, or database was modified for this phase.
