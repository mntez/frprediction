# Backend data services

This service contains the server-side football-data.org v4 client and the
football data persistence/ingestion layer. The provider token and database URL
are loaded from the repository-root `.env` file or from the process environment.
Neither credential belongs in the Next.js frontend.

Install dependencies from `backend`:

```powershell
py -m pip install -r requirements.txt
```

Set these values in the repository-root `.env` file:

```env
FOOTBALL_DATA_API_KEY=your-provider-token
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/frpredictions
```

Apply schema migrations; production tables are not created automatically:

```powershell
py -m alembic upgrade head
```

Import one league, optionally by season start year or date range:

```powershell
py -m app.jobs.ingest --competition PL --season 2025
py -m app.jobs.ingest --competition PL --date-from 2025-08-01 --date-to 2026-06-01
py -m app.jobs.ingest --competition PL --all-seasons
py -m app.jobs.ingest --competition PL --all-seasons --date-from 2025-08-01 --date-to 2026-06-01
```

With no `--competition`, the job synchronizes the six configured leagues. A
season and date range may be combined. football-data.org treats `dateTo` as an
exclusive upper bound for match date ranges.
`--all-seasons` first reads the seasons exposed by `/competitions/{code}`, then
imports each advertised season sequentially. It requires one explicit
competition so a large historical import is scoped deliberately. The account's
available season list determines what can be imported. An unfinished match is
stored as a fixture and does not create a result.

Build the chronological historical training dataset (without training a model):

```powershell
py -m app.jobs.build_training_dataset
```

The processed CSV is written under `ml/data/processed/`; feature definitions,
missing-value rules, and leakage controls are documented in `ml/README.md`.

From the repository root, train and evaluate the versioned Poisson baseline
using the fixed chronological cutoff documented in `ml/README.md`:

```powershell
py -m ml.train_poisson_baseline
```

Train and compare the multinomial Logistic Regression baseline on that same
chronological test set:

```powershell
py -m ml.train_logistic_regression
```

Train and compare the conservative XGBoost 1X2 baseline on the same fixed
chronological test set:

```powershell
py -m ml.train_xgboost
```

The model, training-only preprocessing metadata, per-match predictions, and
comparison report are saved under `ml/models/xgboost/v1/` and
`ml/evaluation/xgboost/v1/`.

Start the FastAPI application separately with:

```powershell
py -m uvicorn app.main:app --reload
```

The provider client is available as `app.state.football_data` while the
application lifespan is active. It exposes `get_competitions`, `get_matches`,
`get_competition_matches`, `get_standings`, and `get_teams`; these are backend
service methods and do not create public application endpoints.

Match date ranges require both `date_from` and `date_to`. Match status and
competition filters, season and matchday filters for competition matches,
season/matchday/date filters for standings, and season filters for teams map
to the corresponding v4 query parameters. football-data.org treats `dateTo` as
an exclusive upper bound for match date ranges.

Competition, season, team, fixture, and result records retain the provider IDs.
Ingestion updates records in per-competition transactions and is safe to rerun.
The provider currently exposes the competition identifiers `PL`, `PD`, `SA`,
`BL1`, `FL1`, and `DED` for the configured initial leagues. Endpoint access and
available seasons/data depend on the football-data.org account plan. HTTP 429
responses are surfaced as rate-limit errors and include `Retry-After` when the
provider supplies it; the client does not automatically retry and risk
exceeding the provider's request limits.
