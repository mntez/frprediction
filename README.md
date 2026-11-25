# FRPREDICTIONS

Football data and prediction platform, organized into separate frontend and backend projects.

```text
frpredictions/
├── frontend/   Next.js website
├── backend/    FastAPI service, ingestion job, and database migrations
├── .env        Local backend credentials (never commit)
└── .env.example
```

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>.

## Backend

Configure `FOOTBALL_DATA_API_KEY` and `DATABASE_URL` in the repository-root `.env`, then from `backend`:

```powershell
py -m pip install -r requirements.txt
py -m alembic upgrade head
py -m uvicorn app.main:app --reload
```

To ingest data, run from `backend`:

```powershell
py -m app.jobs.ingest --competition PL --season 2025
```

See [backend/README.md](backend/README.md) for historical ingestion options.
