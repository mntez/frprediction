# Phase 3G.2.3 — CI Failure Diagnosis

## Scope and outcome

Authenticated GitHub Actions logs were retrieved read-only for run [45554416162](https://github.com/mntez/frprediction/actions/runs/45554416162), repository `mntez/frprediction`, commit `af576671ac7cb6b52b5fe8e30c20243a49abcf97`, branch `codex/publish-frprediction-project`.

Both jobs failed before pytest collected tests. The root cause is the workflow's incomplete test-runtime installation: `backend/tests/conftest.py` imports SQLAlchemy, but the install step does not install SQLAlchemy or `backend/requirements.txt`. No registry test ran; neither symlink test passed, failed, nor skipped. No code, workflow, test, registry, or protected artifact was changed during this diagnosis.

## Commands and runner environments

Both jobs executed the workflow's exact command from the `backend` working directory:

```text
python -m pytest tests/test_model_registry.py -q
```

Runner and setup details from the logs:

| Job | Runner/runtime | Setup before pytest | Pytest process exit |
|---|---|---|---:|
| Ubuntu | Ubuntu 26.04.1, CPython 3.12.16 | Checkout, Python setup, and package installation succeeded; pytest 9.2.0 installed with joblib, NumPy, SciPy, and scikit-learn 1.9.1 | 4 |
| Windows | `windows-latest`, CPython 3.12.10 | Checkout, Python setup, and package installation succeeded; pytest 9.2.0 installed with joblib, NumPy, SciPy, and scikit-learn 1.9.1 | 1 |

## Relevant log output

Ubuntu:

```text
ImportError while loading conftest '/home/runner/work/frprediction/frprediction/backend/tests/conftest.py'.
tests/conftest.py:5: in <module>
    from sqlalchemy import create_engine
E   ModuleNotFoundError: No module named 'sqlalchemy'
##[error]Process completed with exit code 4.
```

Windows:

```text
ImportError while loading conftest 'D:\a\frprediction\frprediction\backend\tests\conftest.py'.
tests\conftest.py:5: in <module>
    from sqlalchemy import create_engine
E   ModuleNotFoundError: No module named 'sqlalchemy'
##[error]Process completed with exit code 1.
```

The logs establish the same missing-package import failure on both operating systems. They show different process exit codes (4 on Ubuntu and 1 on Windows); they do not establish a separate platform-specific failure. Checkout and package installation completed on both runners. This was not a bad test path or malformed pytest command, and no assertion-level test failure occurred.

## Collection and symlink-test status

Pytest stopped while importing `conftest.py`, before normal test collection. The logs contain no collected-test summary and no entries for either security test. Therefore:

| Test | Ubuntu | Windows hosted job |
|---|---|---|
| `test_symlink_outside_model_root_is_rejected` | Not collected or executed; not skipped | Not collected or executed; not skipped |
| `test_symlinked_directory_outside_model_root_is_rejected` | Not collected or executed; not skipped | Not collected or executed; not skipped |

There are no hosted test counts to report because pytest did not reach collection.

## Root cause classification

- **Runner/setup:** The hosted runners and Python setup succeeded.
- **Dependency setup:** Incomplete. The workflow installs a limited package list but omits SQLAlchemy, which is required by the backend test `conftest.py`. The full backend dependency file includes SQLAlchemy.
- **Pytest invocation:** The command and working directory appear as configured in the workflow and were actually invoked. No command-line usage error is shown in the log.
- **Collection:** Blocked by the `conftest.py` import error.
- **Registry assertions/security tests:** Not reached; there is no evidence of a code-level test failure or symlink behavior in this run.
- **Exit status:** Ubuntu reported code 4 and Windows code 1 for the same pre-collection import error. The logs establish those values but provide no separate reason for their numeric difference.

The workflow needs a separately reviewed dependency-setup change before a new run can verify registry behavior. No fix is made or proposed as an implementation in this diagnosis.

## Files inspected

- `.github/workflows/model-registry-security.yml`
- `backend/tests/conftest.py`
- `backend/requirements.txt`
- Both authenticated job logs for run `45554416162`

No tests were rerun. No source, workflow, test, registry metadata, model, dataset, evaluation artifact, or database was modified.
