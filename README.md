# Encore AI Drawing Matcher

Agentic RAG application that matches uploaded handwritten roofing/flashing sketches against Encore master drawings and fills JSON dimension templates.

## Stack

- **Frontend:** Next.js 15, TypeScript, Tailwind CSS
- **Backend:** FastAPI, Ollama Cloud (qwen3-vl vision + gpt-oss text), [uv](https://docs.astral.sh/uv/) package manager
- **Database:** PostgreSQL 16 + pgvector (port **5455**)
- **Cache:** Redis 7 (port **6377**)
- **Infrastructure:** Docker Compose

## Quick start

### 1. Start PostgreSQL + Redis (Docker)

```bash
docker compose up -d
```

| Service | Host port | Connection URL |
|---|---|---|
| PostgreSQL | `5455` | `postgresql://encore:encore@localhost:5455/encore_drawings` |
| Redis | `6377` | `redis://localhost:6377/0` |

Verify containers are healthy:

```bash
docker compose ps
```

### 2. Backend

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
cd backend
uv sync
cp .env.example .env   # add your OLLAMA_API_KEY
uv run uvicorn app.main:app --reload --port 8000
```

Required env vars in `backend/.env`:

```env
OLLAMA_API_KEY=your_key_here
DATABASE_URL=postgresql+asyncpg://encore:encore@localhost:5455/encore_drawings
REDIS_URL=redis://localhost:6377/0
REDIS_CACHE_TTL_SECONDS=86400
```

Health check (confirms Postgres + Redis are connected):

```bash
curl http://localhost:8000/api/v1/health
# { "status": "ok", "postgres": true, "redis": true }
```

On first startup the backend will:
- Create DB tables automatically (including pgvector HNSW index on `master_drawings.embedding`)
- Seed **48 master drawings** from the filesystem into Postgres
- **Backfill embeddings** for any masters missing vectors (Ollama `nomic-embed-text`, cached in Redis)
- Import existing file-based corrections into the `corrections` table

Re-embed all masters manually:

```bash
cd backend && uv run python scripts/backfill_embeddings.py
```

### 3. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open http://localhost:3000

**Detailed flow documentation:** [docs/UPLOAD_TO_RESULTS.md](docs/UPLOAD_TO_RESULTS.md) — upload → pipeline steps → results → debugging

## Usage

1. Upload a handwritten sketch image (PNG/JPG)
2. Click **Match Drawing**
3. View matched master, confidence, side-by-side comparison, and filled JSON
4. Download the filled JSON export
5. If wrong → **Correct this match** → save correction (stored in Postgres + feedback files)

## What is stored where

| Data | PostgreSQL | Redis | Files |
|---|---|---|---|
| Master catalog (48 drawings) | `master_drawings` (+ pgvector embeddings) | — | `training_testing_datasets/Training/` |
| Match job results | `match_jobs` | — | uploads in `backend/data/uploads/` |
| User corrections | `corrections` | — | `training_testing_datasets/feedback/` |
| Ollama vision/compare cache | — | image hash keys (24h TTL) | — |
| Ollama text embed cache | — | embed hash keys (24h TTL) | — |

## Dataset

- Master catalog: `training_testing_datasets/Training/Encore_master_drawings/` (48 PNG + JSON pairs)
- Test sketches: `training_testing_datasets/testing/Client_handwritten_data/`
- Corrections: `training_testing_datasets/feedback/`

## API

- `GET /api/v1/health` — status + postgres/redis connectivity
- `POST /api/v1/match` — upload image, returns match result
- `GET /api/v1/masters` — list master drawings
- `GET /api/v1/masters/{category}/{basename}/image` — master PNG
- `GET /api/v1/match/{job_id}/export` — download filled JSON
- `POST /api/v1/feedback` — save a correction `{ job_id, master_key, lengths[], note? }`
- `GET /api/v1/feedback` — list saved corrections

## Phase 2: Feedback / Training

When a match is wrong:

1. Click **Correct this match** on the results panel
2. Select the correct master from the dropdown
3. Edit segment lengths if needed
4. Click **Save correction & train**

Corrections are saved to:
- **PostgreSQL** `corrections` table (durable, queryable)
- **Files** under `training_testing_datasets/feedback/` (`images/`, `labels/`, `manifest.jsonl`)

Future similar sketches get a retrieval boost toward corrected masters.

## Backend dependency management

Dependencies are defined in [`backend/pyproject.toml`](backend/pyproject.toml) and locked in [`backend/uv.lock`](backend/uv.lock).

```bash
cd backend
uv add <package>      # add a dependency
uv sync               # install from lockfile
uv run <command>      # run in project venv
```

## Docker commands

```bash
docker compose up -d      # start Postgres + Redis
docker compose down       # stop containers
docker compose logs -f    # view logs
docker compose ps         # check health status
```

Stop and remove volumes (wipes DB data):

```bash
docker compose down -v
```
