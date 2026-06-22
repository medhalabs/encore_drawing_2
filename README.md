# Encore AI Drawing Matcher

Matches handwritten roofing/flashing sketches against Encore master drawings and fills JSON dimension templates.

**How it works:**
1. **EfficientNet-B0 classifier** (fast path) — identifies the master shape in <1s. If confidence ≥ 85%, skips the LLM compare entirely.
2. **LLM vision compare** (fallback) — used when classifier is unsure or shape is new.
3. **LLM length extraction** (always) — reads the handwritten mm numbers off the sketch.

The classifier improves automatically: every 10 user corrections trigger a background retrain.

## Stack

- **Frontend:** Next.js 15, TypeScript, Tailwind CSS
- **Backend:** FastAPI, Python, [uv](https://docs.astral.sh/uv/) package manager
- **Classifier:** PyTorch EfficientNet-B0 (incremental / online retraining)
- **Vision LLM:** Ollama Cloud — `gemma4:31b-cloud` (shape compare + length extraction)
- **Database:** PostgreSQL 16 + pgvector (port **5455**)
- **Cache:** Redis 7 (port **6377**)
- **Infrastructure:** Docker Compose

## Quick start

### 1. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Postgres + Redis)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)
- Node.js 18+

### 2. Start infrastructure (PostgreSQL + Redis)

```bash
docker compose up -d
docker compose ps     # wait until both show "(healthy)"
```

| Service    | Host port | URL                                                       |
|------------|-----------|-----------------------------------------------------------|
| PostgreSQL | `5455`    | `postgresql://encore:encore@localhost:5455/encore_drawings` |
| Redis      | `6377`    | `redis://localhost:6377/0`                                |

### 3. Backend

```bash
cd backend
uv sync
cp .env.example .env   # then fill in your OLLAMA_API_KEY
uv run uvicorn app.main:app --reload --port 8000
```

Required vars in `backend/.env`:

```env
OLLAMA_API_KEY=your_key_here
OLLAMA_VISION_MODEL=gemma4:31b-cloud
DATABASE_URL=postgresql+asyncpg://encore:encore@localhost:5455/encore_drawings
REDIS_URL=redis://localhost:6377/0
```

**What happens on first startup:**
- DB tables created automatically (pgvector HNSW index included)
- **96 master drawings** seeded (48 originals + 48 left↔right mirror variants)
- Embeddings backfilled via `nomic-embed-text` (cached in Redis)
- EfficientNet classifier trains in the background on the master images (takes ~30s, runs in a background thread — the API is usable immediately)
- Existing feedback corrections imported

Health check:

```bash
curl http://localhost:8000/api/v1/health
# { "status": "ok", "postgres": true, "redis": true }
```

### 4. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Open http://localhost:3000

---

## Usage

1. Upload a handwritten sketch (PNG/JPG)
2. Click **Match Drawing**
3. View matched master, confidence, side-by-side comparison, and filled JSON
4. Download the filled JSON export
5. If wrong → **Correct this match** → select the right master + edit lengths → **Save**

Every 10 corrections automatically retrain the EfficientNet classifier in the background (no restart needed).

---

## Master catalog

| Location | Contents |
|---|---|
| `training_testing_datasets/Training/Encore_master_drawings/` | 48 original PNG + JSON pairs |
| Same directories, `*-mirror.png` files | 48 left↔right flipped variants (auto-generated) |
| `backend/data/models/efficientnet_v*.pt` | Versioned classifier weights |

Categories: Aprons, Capping, FootMoulds, Gutters, Misc, RidgeValley, Soakers

To regenerate mirror variants:

```bash
cd backend
uv run python3 -c "
from pathlib import Path
from PIL import Image
root = Path('../training_testing_datasets/Training/Encore_master_drawings')
for src in root.rglob('*.png'):
    if '-mirror' not in src.stem:
        dest = src.parent / f'{src.stem}-mirror.png'
        if not dest.exists():
            Image.open(src).transpose(Image.FLIP_LEFT_RIGHT).save(dest)
            print(dest)
"
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Status + postgres/redis connectivity |
| `POST` | `/api/v1/match` | Upload sketch → match result |
| `GET` | `/api/v1/masters` | List all master drawings |
| `GET` | `/api/v1/masters/{category}/{basename}/image` | Master PNG |
| `GET` | `/api/v1/match/{job_id}/export` | Download filled JSON |
| `POST` | `/api/v1/feedback` | Save a correction `{ job_id, master_key, lengths[], note? }` |
| `GET` | `/api/v1/feedback` | List saved corrections |

---

## Data storage

| Data | PostgreSQL | Redis | Files |
|------|------------|-------|-------|
| Master catalog (96 drawings) | `master_drawings` + pgvector embeddings | — | `training_testing_datasets/Training/` |
| Match job results | `match_jobs` | — | `backend/data/uploads/` |
| User corrections | `corrections` | — | `training_testing_datasets/feedback/` |
| Classifier weights | — | — | `backend/data/models/efficientnet_v*.pt` |
| Vision/compare LLM cache | — | image hash keys (24h TTL) | — |
| Text embed cache | — | embed hash keys (24h TTL) | — |

---

## EfficientNet classifier

The classifier starts cold (low confidence) and improves over time as corrections accumulate.

| Stage | What happens |
|---|---|
| Startup | Trains on 48 master originals in background (~30s) |
| Every 10 corrections | Automatic retrain on masters + all corrections |
| confidence ≥ 85% | LLM compare skipped — only LLM length extraction runs |
| confidence < 85% | Full LLM path: analyze → retrieve → compare → extract |

Force a retrain manually:

```bash
cd backend
uv run python3 -c "
from app.main import catalog, classifier, retrain_service, _model_dir
catalog.load()
all_keys = [m.key for m in catalog.masters]
classifier._label_index = sorted(all_keys)
classifier._key_to_idx = {k: i for i, k in enumerate(classifier._label_index)}
retrain_service.retrain_now()
import time; time.sleep(60)   # wait for background thread
print('Done:', list(_model_dir.glob('*.pt')))
"
```

---

## Backend dependency management

```bash
cd backend
uv add <package>    # add dependency
uv sync             # install from lockfile
uv run <command>    # run in project venv
```

---

## Docker commands

```bash
docker compose up -d        # start Postgres + Redis
docker compose ps           # check health
docker compose logs -f      # tail logs
docker compose down         # stop containers
docker compose down -v      # stop + wipe DB volumes
```
