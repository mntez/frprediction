# Phase 3G.2.4 — CI Test Dependency Fix

## Change

Updated `.github/workflows/model-registry-security.yml` to install backend and test dependencies from the project's authoritative dependency file:

```text
working-directory: backend
python -m pip install -r requirements.txt
```

This replaces the workflow's incomplete hand-maintained package list. `backend/requirements.txt` includes SQLAlchemy and the test dependencies, and `backend/README.md` documents this as the standard backend installation method. The Ubuntu/Windows matrix and Python 3.12 configuration remain unchanged. Registry code and tests were not modified.

## Local verification

On this Windows development environment, from `backend`:

```text
py -m pytest tests/test_model_registry.py -q
```

Result: **14 passed, 2 skipped**. Both symlink tests skipped because local Windows does not permit their creation. The suite successfully imported `conftest.py` and completed collection/execution for the other tests locally. This does not verify Linux symlink behavior.

## Hosted verification

**Pending.** The corrected workflow has not yet completed on GitHub. Do not treat dependency installation or Ubuntu symlink execution as verified until the new run completes successfully and the Ubuntu log explicitly reports both:

- `test_symlink_outside_model_root_is_rejected` — passed.
- `test_symlinked_directory_outside_model_root_is_rejected` — passed.

Record the new run URL, commit SHA, per-OS test counts, and both symlink outcomes after completion. The prior failed run (`45554416162`) predates this dependency fix and is not evidence for the corrected workflow.

## Scope

Only the workflow and this report are changed for Phase 3G.2.4. No model training, prediction generation, activation, PostgreSQL change, migration, protected model/dataset/evaluation artifact change, or registry implementation change occurred.
