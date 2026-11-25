# AGENTS.md — Football AI Prediction Platform

## 1. Project Mission

This repository contains a football prediction platform powered by a custom machine-learning prediction engine.

The product has two equally important parts:

1. A production-quality football prediction website.
2. A data-driven prediction system that produces probabilities for match outcomes and goal markets.

The system should eventually support:

- 1X2 predictions
- Over/Under goals
- Both Teams To Score (BTTS)
- Predicted score
- Confidence/probability scores
- Historical model performance
- Prediction-vs-result tracking
- Multiple leagues
- Optional comparison/consensus with external prediction sources where legally and technically permitted

Initial leagues:

- English Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1
- Eredivisie

The project is NOT intended to copy PredictZ, Forebet, or another prediction site. External prediction sites may be used as comparison/consensus sources only when their terms, licensing, and technical access permit it.

---

## 2. Core Principles

### Inspect before implementing

Before changing code:

1. Inspect the repository structure.
2. Read relevant existing files.
3. Identify the current architecture.
4. Check existing dependencies and scripts.
5. Check `AGENTS.md` and any more specific agent instructions.
6. Explain briefly what you intend to change before making substantial changes.

Do not rebuild existing functionality unnecessarily.

### Build only what was requested

Do not introduce unrelated features, refactors, libraries, or architectural changes without a clear reason.

If a requested change requires an architectural decision, explain the decision before implementing it.

### Test before handoff

Every meaningful implementation must be tested before being considered complete.

At minimum:

- Run relevant unit tests.
- Run type checking for frontend changes.
- Run linting where configured.
- Run backend tests for API changes.
- Run ML evaluation/tests for model changes.
- Verify imports and application startup.
- Report what was tested and any remaining issues.

Never claim something works if it was not tested.

---

## 3. Technology Stack

### Frontend

- Next.js
- TypeScript
- Tailwind CSS

### Backend

- Python
- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL

### Machine Learning

- Python
- Pandas
- NumPy
- SciPy
- scikit-learn
- XGBoost
- Poisson/statistical models
- Joblib for initial model persistence

### Infrastructure

- Git
- GitHub
- Docker/Docker Compose when useful
- Managed PostgreSQL in production
- Vercel for the frontend when appropriate
- Render, Railway, Fly.io, or equivalent for backend/workers

### Background processing

Start simple with APScheduler or an equivalent scheduler.

Introduce Redis and a queue only when the workload requires it.

---

## 4. Repository Structure

Prefer a monorepo structure similar to:

```text
football-ai/
├── apps/
│   └── web/
│       ├── app/
│       ├── components/
│       ├── lib/
│       ├── public/
│       └── package.json
│
├── services/
│   ├── api/
│   │   ├── app/
│   │   │   ├── api/
│   │   │   ├── core/
│   │   │   ├── models/
│   │   │   ├── schemas/
│   │   │   ├── services/
│   │   │   └── main.py
│   │   └── requirements.txt
│   │
│   └── worker/
│       ├── jobs/
│       └── main.py
│
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── external/
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── models/
│
├── database/
│   ├── migrations/
│   └── seeds/
│
├── scripts/
│   ├── fetch_data.py
│   ├── prepare_data.py
│   └── train_model.py
│
├── tests/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── README.md
└── AGENTS.md
```

Do not force this exact structure if the repository already has a better established structure. Adapt to the existing project rather than restructuring for its own sake.

---

## 5. Separation of Responsibilities

Keep these boundaries clear:

### Frontend

Responsible for:

- UI
- routing
- presentation
- user interaction
- API consumption

The frontend must NOT contain:

- ML training logic
- prediction algorithms
- secret API keys
- direct database credentials
- football-provider secret credentials

### FastAPI

Responsible for:

- API endpoints
- validation
- authentication if introduced later
- database access
- prediction requests
- application/business logic

### ML

Responsible for:

- dataset preparation
- feature engineering
- model training
- model evaluation
- model versioning
- inference

### Worker

Responsible for:

- scheduled data ingestion
- scheduled feature updates
- scheduled prediction generation
- post-match result processing

### PostgreSQL

Responsible for persistent application and prediction data.

---

## 6. Data Rules

Football data is the foundation of this project.

Never invent:

- fixtures
- results
- team statistics
- player statistics
- injuries
- xG
- odds
- predictions
- historical records

If real data is unavailable, explicitly use mock/sample data and label it as such.

External data must have a documented source.

Store useful source metadata where appropriate:

```text
source
source_id
retrieved_at
```

Respect the terms, licensing, rate limits, and access rules of every external data provider.

Do not assume that because information is publicly visible on a website it can automatically be scraped, redistributed, or commercially reused.

---

## 7. API Keys and Secrets

Never hard-code secrets.

Use environment variables.

Example:

```env
DATABASE_URL=
FOOTBALL_API_KEY=
REDIS_URL=
```

Maintain:

```text
.env
```

locally and keep it out of Git.

Maintain:

```text
.env.example
```

with variable names but no real secrets.

Never print API keys, database passwords, tokens, or private credentials in logs.

Never commit secrets.

---

## 8. Machine Learning Rules

The prediction system must be reproducible and measurable.

### Ground truth

Actual historical match results are the primary ground truth.

Do NOT train the primary model by treating PredictZ, Forebet, or another prediction site's forecast as the truth.

External predictions may later be used as comparison features or consensus signals if legally and technically permitted.

### Data leakage

This is one of the most important rules in the project.

A feature used to predict a match must only contain information that would have been available BEFORE that match started.

Never use:

- final match statistics
- post-match xG
- post-match standings
- future results
- future team form
- future player information

when constructing pre-match features.

Any feature that could leak future information must be rejected or carefully time-aligned.

### Time-based validation

Do not randomly shuffle historical football matches for the primary evaluation.

Prefer chronological/time-based splits.

Example:

```text
TRAIN
2018–2024

VALIDATION
2025

TEST
2026
```

The exact dates depend on the available dataset.

### Baselines

Always establish simple baselines before complex models.

Initial candidates:

- Home/draw/away baseline
- Logistic Regression
- Poisson model
- Random Forest if useful
- XGBoost

Neural networks should only be introduced when there is enough data and a demonstrated reason to use them.

Do not use a neural network simply because it sounds more advanced.

---

## 9. Prediction Outputs

The model should primarily produce probabilities rather than only a hard prediction.

Example:

```json
{
  "home_win": 0.613,
  "draw": 0.227,
  "away_win": 0.160,
  "over_1_5": 0.840,
  "over_2_5": 0.630,
  "over_3_5": 0.370,
  "btts": 0.580
}
```

The system may additionally produce:

- predicted home goals
- predicted away goals
- predicted score
- confidence/calibration information

Do not represent model probability as certainty.

Avoid language such as:

- "guaranteed win"
- "sure bet"
- "100% prediction"

unless clearly describing a mathematical property rather than claiming certainty about a football result.

---

## 10. Model Evaluation

Accuracy alone is insufficient.

Track appropriate metrics for each task.

For classification:

- Accuracy
- Log Loss
- Brier Score
- Precision/Recall where appropriate
- Calibration

For goal prediction:

- MAE
- RMSE
- Poisson/deviance-style metrics where appropriate

For market predictions:

- Over 1.5 accuracy
- Over 2.5 accuracy
- Over 3.5 accuracy
- BTTS accuracy

Always evaluate against a clearly defined test period.

Never change the test set simply because a model performs poorly on it.

---

## 11. Model Versioning

Every production prediction must be associated with a model version.

Example:

```text
model_version: v1.0.0
training_data_start: 2018-08-01
training_data_end: 2026-06-01
feature_count: 48
created_at: ...
```

Never silently replace a production model.

When a new model is trained:

1. Evaluate it.
2. Compare it against the current production model.
3. Record the evaluation.
4. Only promote it when the criteria are met.
5. Record the new model version.

---

## 12. Prediction Lifecycle

The intended lifecycle is:

```text
Football API
      ↓
Data ingestion
      ↓
Validation
      ↓
PostgreSQL
      ↓
Feature generation
      ↓
Model inference
      ↓
Prediction stored
      ↓
Website
      ↓
Match finishes
      ↓
Actual result stored
      ↓
Prediction evaluated
      ↓
Performance metrics updated
```

Predictions should be stored before kickoff whenever possible.

Do not overwrite historical predictions after a result is known.

This is necessary for honest performance tracking.

---

## 13. API Design

Prefer clear resource-oriented endpoints.

Examples:

```text
GET /api/v1/leagues
GET /api/v1/leagues/{league_id}/matches
GET /api/v1/matches/today
GET /api/v1/matches/{match_id}
GET /api/v1/matches/{match_id}/prediction
GET /api/v1/predictions/today
GET /api/v1/model-performance
```

Keep API contracts typed and documented.

Use Pydantic schemas for request/response validation.

Do not expose database models directly when a dedicated API schema is more appropriate.

---

## 14. Database Principles

Use migrations for schema changes.

Do not manually modify production database schemas without a migration.

Important entities may include:

```text
leagues
seasons
teams
matches
team_match_stats
players
player_match_stats
injuries
predictions
prediction_results
model_versions
data_sources
```

Use stable IDs and source IDs where appropriate.

Prevent duplicate fixtures and duplicate data ingestion.

---

## 15. Design System

### Brand color palette

Use the following palette as the shared visual foundation for the website. Keep these values consistent across pages and components; do not introduce competing brand colors without a clear product need.

| Name | Hex | Suggested use |
|---|---|---|
| Vinho | `#69212C` | Primary brand color, key actions, selected states, and strong accents |
| Gelo | `#DCE0E9` | Light surfaces, panels, and backgrounds |
| Névoa | `#8CA2AD` | Secondary accents, muted information, and supporting UI |
| Aço | `#27353E` | Main dark color, text on light surfaces, and dark backgrounds |

Check text and control contrast when applying the palette, and preserve readable focus, hover, and disabled states.

### Typography

Use two complementary font families consistently throughout the design system:

- **Display font:** `DM Serif Display` for prominent page titles and selected editorial headings.
- **Interface font:** `Inter` for body text, navigation, forms, buttons, tables, and match data.

Load both through the project's Next.js font setup (prefer `next/font/google`, or local font files when supplied) and expose them as shared design tokens/CSS variables. Avoid adding extra font families without a clear design need.

---

## 16. Frontend Principles

The UI should be:

- clean
- fast
- responsive
- data-focused
- easy to understand

The product should not overwhelm users with raw statistics.

A match card should communicate the important information immediately:

```text
Team A vs Team B

AI Prediction
Team A 61%
Draw 23%
Team B 16%

Predicted score
2–1

Over 2.5
63%

BTTS
58%

Model confidence
78/100
```

Detailed statistics can live on the match details page.

Never display fabricated confidence.

If confidence is derived from calibration or another defined methodology, document it.

---

## 17. External Prediction Sources

PredictZ and Forebet may be used as comparison sources only where permitted.

Possible future display:

```text
OUR MODEL
Arsenal

PredictZ
Arsenal

Forebet
Arsenal

Consensus
3/3
```

Do not copy their branding, UI, text, proprietary algorithms, or protected content.

Do not claim their predictions are part of our model.

Clearly distinguish:

```text
Our Model
External Source
Consensus
```

Any automated collection must comply with the relevant site's terms and technical restrictions.

---

## 18. Testing

### Frontend

Test:

- components
- API integration
- loading states
- error states
- responsive layouts

### Backend

Test:

- API endpoints
- validation
- database operations
- error handling
- authentication if introduced

### ML

Test:

- feature calculations
- no future leakage
- dataset splits
- model training
- inference
- probability sums
- evaluation metrics

Example invariant:

```text
home_probability + draw_probability + away_probability ≈ 1
```

### Data ingestion

Test:

- duplicate prevention
- missing values
- invalid team IDs
- invalid fixtures
- provider API failures
- rate-limit handling

---

## 19. Error Handling

External APIs can fail.

The system must handle:

- timeout
- HTTP errors
- rate limits
- malformed responses
- missing data
- partial data

Do not silently treat failed API requests as valid empty datasets.

Log enough information to diagnose failures without exposing secrets.

---

## 20. Development Workflow

For every task:

### Step 1 — Inspect

Understand the current code.

### Step 2 — Plan

State briefly:

- what will change
- which files are affected
- why
- how it will be tested

### Step 3 — Implement

Make the smallest clean change that satisfies the request.

### Step 4 — Test

Run relevant tests/checks.

### Step 5 — Review

Check:

- regressions
- unused dependencies
- secrets
- type errors
- lint errors
- data leakage
- unnecessary complexity

### Step 6 — Report

Summarize:

- what changed
- what was tested
- any remaining issues

---

## 21. Dependency Rules

Do not add a package just because it is convenient.

Before adding a dependency:

1. Check whether the existing stack already solves the problem.
2. Check maintenance/activity.
3. Check compatibility.
4. Check license.
5. Check security/reputation.
6. Explain why it is needed if the dependency materially changes the architecture.

Keep dependencies minimal.

---

## 22. Security

Never:

- commit secrets
- expose database credentials
- expose football API keys to the browser
- trust client-supplied prediction results
- directly interpolate untrusted SQL
- disable security checks merely to make development easier

Use environment variables and server-side API access.

Validate all external data.

---

## 23. Performance

Do not fetch the same football data repeatedly for every page request.

Prefer:

```text
Provider API
      ↓
Scheduled ingestion
      ↓
Database/cache
      ↓
FastAPI
      ↓
Next.js
```

The website should primarily read from our backend/database rather than repeatedly calling external providers.

Cache appropriate read-heavy endpoints.

---

## 24. Deployment

Development should work locally before deployment.

Production architecture may eventually be:

```text
Vercel
  ↓
Next.js

Backend host
  ↓
FastAPI

Worker host
  ↓
Scheduled jobs

Managed PostgreSQL
  ↓
Database

Redis
  ↓
Cache/queue when needed
```

Do not introduce production infrastructure before the local development workflow is stable.

---

## 25. Environment Setup

Expected local tools:

```text
Node.js
Python 3.x
Git
PostgreSQL
VS Code
```

Recommended initial Python environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Backend dependencies should be recorded in the project's dependency file.

Do not rely on globally installed Python packages.

---

## 26. AI Coding Agent Behavior

The coding agent must:

- inspect before editing
- preserve working code
- avoid unnecessary rewrites
- explain substantial plans before implementation
- create files only when needed
- use existing components/utilities where appropriate
- test changes
- never invent API responses
- never fabricate football data
- never fabricate model accuracy
- never claim an ML model is accurate without evaluation
- never introduce future-data leakage
- document important ML decisions
- keep the codebase maintainable

If requirements are ambiguous, identify the ambiguity instead of silently making a major architectural assumption.

---

## 27. Definition of Done

A feature is not complete until:

- implementation is finished
- relevant tests pass
- type/lint checks pass where applicable
- the application starts successfully
- no secrets were committed
- documentation is updated when necessary
- no unrelated files were changed
- the final response clearly states what was changed and tested

For ML features, additionally:

- training/evaluation methodology is documented
- data leakage was considered
- test data remains isolated
- metrics are recorded
- model version is identifiable

---

## 28. Current Development Priority

Build in this order unless the user explicitly changes the plan:

1. Repository/project foundation
2. Next.js frontend
3. FastAPI backend
4. PostgreSQL connection
5. Football data provider integration
6. Historical data ingestion
7. Data validation and normalization
8. Feature engineering
9. Baseline prediction models
10. Time-based model evaluation
11. Prediction API
12. Prediction UI
13. Automated prediction generation
14. Prediction-result tracking
15. Model performance dashboard
16. External prediction-source comparison
17. Model improvement
18. Production deployment
19. Monitoring and scheduled retraining

Do not skip directly to advanced AI before the data pipeline and evaluation framework are reliable.
