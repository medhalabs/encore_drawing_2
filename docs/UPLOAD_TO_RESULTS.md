# Upload to Results — End-to-End Flow

This document explains what happens from the moment you upload a handwritten sketch in the UI until match results are displayed, including every pipeline step, API call, model invocation, and data store involved.

---

## Table of contents

1. [High-level overview](#high-level-overview)
2. [Architecture diagram](#architecture-diagram)
3. [Frontend flow](#frontend-flow)
4. [API layer](#api-layer)
5. [Pipeline steps (backend)](#pipeline-steps-backend)
6. [Scoring and decision logic](#scoring-and-decision-logic)
7. [Persistence and caching](#persistence-and-caching)
8. [Results UI](#results-ui)
9. [Correction / training flow](#correction--training-flow)
10. [Debugging with the live pipeline](#debugging-with-the-live-pipeline)
11. [Configuration reference](#configuration-reference)
12. [Typical timing](#typical-timing)

---

## High-level overview

```
User uploads PNG/JPG
       ↓
Next.js sends multipart POST → /api/v1/match/stream (SSE)
       ↓
FastAPI saves file → runs 7-step agentic pipeline
       ↓
Each step streams back to UI in real time
       ↓
Final MatchResult → results panel + Postgres + optional Redis cache
```

The system does **not** fine-tune models at runtime. It:

1. **Analyzes** the sketch with a vision model (Ollama Cloud `qwen3-vl`)
2. **Retrieves** candidate master drawings from a catalog of 48 templates
3. **Compares** sketch vs master PNGs with the same vision model
4. **Selects** the best master and **extracts** handwritten dimension values
5. **Fills** the master JSON template with those lengths
6. **Returns** the filled Encore JSON plus debug trace

---

## Architecture diagram

```mermaid
sequenceDiagram
    participant User
    participant NextJS as Next.js UI
    participant API as FastAPI
    participant Redis
    participant PG as PostgreSQL
    participant Ollama as Ollama Cloud
    participant Catalog as Master Catalog

    User->>NextJS: Drop/select sketch image
    User->>NextJS: Click Match Drawing
    NextJS->>API: POST /api/v1/match/stream (multipart file)

    API->>API: save_upload → backend/data/uploads/{job_id}.png
    API-->>NextJS: SSE event: step upload

    API->>Redis: cache get (vision key)
    alt cache miss
        API->>Ollama: analyze sketch (qwen3-vl)
        API->>Redis: cache set
    end
    API-->>NextJS: SSE event: step analyze

    API->>Catalog: score 48 masters + feedback boosts
    API-->>NextJS: SSE event: step retrieve

    loop top 5 candidates
        API->>Ollama: compare sketch vs master PNG
    end
    API-->>NextJS: SSE event: step compare

    API->>API: select best master + score breakdown
    API-->>NextJS: SSE event: step match

    API->>Ollama: extract lengths for N segments
    API-->>NextJS: SSE event: step extract

    API->>API: validate JSON schema
    API-->>NextJS: SSE event: step validate

    API->>PG: INSERT match_jobs
    API-->>NextJS: SSE event: result (full MatchResult)

    NextJS->>User: Show results panel
```

---

## Frontend flow

### Files involved

| File | Role |
|---|---|
| [`frontend/app/page.tsx`](../frontend/app/page.tsx) | Main page: upload, match button, state |
| [`frontend/components/ImageUpload.tsx`](../frontend/components/ImageUpload.tsx) | Drag-and-drop + preview |
| [`frontend/components/MatchProgress.tsx`](../frontend/components/MatchProgress.tsx) | Live debug pipeline panel |
| [`frontend/components/MatchResults.tsx`](../frontend/components/MatchResults.tsx) | Final results, scores, JSON export |
| [`frontend/lib/api.ts`](../frontend/lib/api.ts) | `matchDrawingStream()` SSE client |

### Step-by-step (browser)

1. **Upload** — User selects a file via `ImageUpload`. A local object URL preview is shown. File is held in React state (`file`).

2. **Match Drawing** — `handleMatch()` in `page.tsx`:
   - Clears previous result and trace
   - Calls `matchDrawingStream(file, handlers)` from `lib/api.ts`
   - Sets `loading = true` and shows the pipeline panel immediately

3. **SSE streaming** — `matchDrawingStream` POSTs to:
   ```
   POST http://localhost:8000/api/v1/match/stream
   Content-Type: multipart/form-data
   body: file=<image bytes>
   ```
   Response is `text/event-stream`. The client reads chunks and parses events:
   - `event: step` → updates live pipeline (`onStep`)
   - `event: result` → sets final `MatchResult` (`onResult`)
   - `event: error` → shows error message (`onError`)

4. **Live pipeline UI** — Each `step` event appends to `liveTrace`. `MatchProgress` shows:
   - Step name, status, message, timestamp
   - Expandable JSON debug payload per step
   - "Show raw JSON" toggle for full trace

5. **Results** — When `event: result` arrives, `MatchResults` renders:
   - Matched master name and confidence
   - Score breakdown (retrieval / vision / feedback / combined)
   - Side-by-side sketch vs master PNG
   - Dimension table (template vs extracted)
   - Filled JSON preview + Download button

---

## API layer

### Primary endpoint (used by UI)

```
POST /api/v1/match/stream
```

| Input | Multipart field `file` — PNG/JPG/WebP, max 10 MB |
| Output | Server-Sent Events stream |
| Events | `step`, `result`, `error` |

**SSE event format:**

```
event: step
data: {"step":"analyze","status":"completed","message":"...","data":{...}}

event: result
data: {"job_id":"...","matched_master":{...},"filled_json":{...}, ...}

event: error
data: {"detail":"error message"}
```

### Legacy sync endpoint

```
POST /api/v1/match
```

Returns the full `MatchResult` JSON in one response (no live steps). The UI no longer uses this by default.

### Supporting endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/match/{job_id}/upload` | Serve uploaded sketch for side-by-side view |
| `GET /api/v1/masters/{category}/{basename}/image` | Serve master machine drawing PNG |
| `GET /api/v1/match/{job_id}/export` | Download filled JSON file |
| `POST /api/v1/feedback` | Save user correction |
| `GET /api/v1/health` | Check Postgres + Redis connectivity |

### Backend entry point

[`backend/app/api/routes/match.py`](../backend/app/api/routes/match.py):

1. Validates file size
2. Calls `MatchService.save_upload()` → writes `backend/data/uploads/{uuid}.png`
3. Runs `MatchService.process_match_stream()` and yields SSE events

Orchestration lives in [`backend/app/services/match_service.py`](../backend/app/services/match_service.py) and [`backend/app/features/agent/orchestrator.py`](../backend/app/features/agent/orchestrator.py).

---

## Pipeline steps (backend)

Each step emits an `AgentTraceStep` streamed to the UI.

### Step 0: `upload`

**What happens:** File saved to disk with a UUID `job_id`.

**Output data:**
```json
{
  "job_id": "a9351db0-868d-438a-92c5-590b2b3c0d27",
  "filename": "sketch.png"
}
```

**Disk path:** `backend/data/uploads/{job_id}.png`

---

### Step 1: `analyze`

**Module:** `features/vision/sketch_analyzer.py`  
**Model:** Ollama Cloud `qwen3-vl:235b-instruct`  
**Cache:** Redis key = hash(prompt + image bytes), TTL 24h

**What happens:** Vision model reads the sketch and returns structured JSON:

| Field | Meaning |
|---|---|
| `segment_count` | Number of straight segments in the profile |
| `angles_estimate` | Bend angles between segments (best effort) |
| `handwritten_lengths` | Dimension numbers visible on the sketch |
| `part_class_hint` | e.g. Aprons, Gutters, Cappings |
| `confidence` | Model confidence 0–1 |

**Debug tip:** If `part_class_hint` is wrong (e.g. "Capping" instead of "Aprons"), retrieval may rank incorrect masters.

---

### Step 2: `retrieve`

**Modules:** `features/embeddings/service.py`, `features/rag/vector_retriever.py`, `features/rag/retriever.py`  
**Models:** Ollama `nomic-embed-text` (text embeddings) + existing vision for feedback image compare  
**Storage:** Postgres pgvector cosine search (top 20), Redis embed cache (24h)

**What happens:**

1. **Embed sketch** — Build text from analyze output (part class, segment count, angles, shape description) and embed via Ollama. Cached in Redis.

2. **pgvector search** — Query Postgres for top 20 masters by cosine similarity on stored master embeddings.

3. **Feedback image matching** — Compare upload against saved correction images. If similarity ≥ 72%, add up to +60 score boost for that master key.

4. **Hybrid structural + vector scoring** — Score all 48 masters:

   **Base rule signals:**

   | Signal | Points |
   |---|---|
   | Exact segment count match | +40 |
   | Segment count off by 1 | +10 |
   | Angle distance (closer = more) | up to +25 |
   | Part class hint match | up to +15 |

   **Hybrid merge** (when pgvector is available):

   ```
   score = 0.35 × vector_sim × 100 + 0.65 × rule_score + feedback_boosts − wrong_master_penalty
   ```

   | Extra signal | Points |
   |---|---|
   | Saved correction (same master + segment count) | +50 |
   | Previously corrected-away master | −35 |

5. Return **top 5** candidates.

**Debug data example:**
```json
{
  "vector_top_candidates": [
    {"key": "Aprons/apron-6", "similarity": 0.842},
    {"key": "Capping/capping-9", "similarity": 0.711}
  ],
  "sketch_embed_preview": "part_class=Aprons | segments=4 | angles=[90,90,90] | description=...",
  "candidates": [
    {"key": "Aprons/apron-6", "score": 85, "reasons": ["segment_count_match", "vector_sim=0.84", "feedback_segment_match:ddf9a591"]},
    {"key": "Capping/capping-9", "score": 72, "reasons": ["segment_count_match", "vector_sim=0.71", "part_class_match"]}
  ],
  "feedback_image_boosts": {"Aprons/apron-6": 54.0}
}
```

**Note:** pgvector uses **text embeddings** of profile metadata/description, not raw pixels. True shape matching still happens in step 3 (`compare`) via vision model side-by-side PNG comparison.

---

### Step 3: `compare`

**Module:** `features/vision/profile_comparator.py`  
**Model:** Ollama Cloud `qwen3-vl:235b-instruct` (side-by-side: sketch + master PNG)

**What happens:** All **5** retrieval candidates are compared visually. The prompt requires **same profile topology** (U-shape vs L-shape matters). If `same_topology: false`, score is capped at 0.35.

**Combined score per candidate:**
```
combined = retrieval_score / 100 × 0.35 + vision_score × 0.65
```

**Debug data:** List of comparisons with `master_key`, `combined_score`, and reasoning string including raw retrieval and vision scores.

---

### Step 4: `match`

**Module:** `features/agent/orchestrator.py` → `select_master()`

**What happens:**
- Pick candidate with highest combined score from step 3
- Build `ScoreBreakdown`: retrieval_score, vector_score, vision_score, feedback_boost, combined_score
- If `vision_score < 0.65` (configurable), add warning: *"Low shape match — please verify"*

**Output:** Selected `master_key`, confidence, score breakdown.

---

### Step 5: `extract`

**Module:** `features/vision/sketch_analyzer.py` → `extract_lengths()`

**What happens:** Vision model re-reads dimension numbers from the sketch, aligned to the selected master's segment count. Falls back to initial `handwritten_lengths` from step 1 if re-read fails.

**Output:** `extracted_lengths` array, e.g. `[10, 50, 100, 10]`

---

### Step 6: `validate`

**Module:** `features/matching/validator.py`

**What happens:** Validates Encore JSON rules:
- `len(lengths) == len(angles) + 1`
- All lengths > 0

Warnings are included in the final response if validation fails partially.

---

### Final assembly

**Module:** `features/matching/json_filler.py`

**What happens:**
1. Load matched master JSON template (angles, direction, folds, `_id`, etc.)
2. Replace only the `lengths` array with extracted values
3. Build `MatchResult` object
4. Save to in-memory cache, **PostgreSQL** `match_jobs` table
5. Stream `event: result` to frontend

**Example filled JSON** (master template + new lengths):
```json
{
  "_id": "69a11b18e52337e366564bdd",
  "lengths": [10, 50, 100, 10],
  "angles": [-30, 90, -30],
  "partClass": "Aprons",
  "direction": "Right"
}
```

---

## Scoring and decision logic

### Why wrong matches can still happen

| Cause | What to check in debug UI |
|---|---|
| Wrong `part_class_hint` in analyze step | Step 1 data → `part_class_hint` |
| Segment count collision (many masters have 4 segments) | Step 2 → multiple high scores |
| Vision compare too lenient before Phase 3 fixes | Step 3 → vision score still high for wrong shape |
| Correct master not in top 5 retrieval | Step 2 → your expected master missing from candidates |

### Confidence shown in UI

- **Green badge (≥ 65% vision):** `vision_score` drives confidence
- **Amber badge (< 65% vision):** Confidence capped; user should verify or use **Correct this match**

---

## Persistence and caching

### PostgreSQL (port 5455)

| Table | Written when | Contents |
|---|---|---|
| `master_drawings` | App startup seed | 48 master templates + metadata |
| `match_jobs` | After each match | Full result, trace, scores, filled JSON |
| `corrections` | User saves correction | Master key, lengths, note, image path |

### Redis (port 6377)

| Key pattern | Purpose | TTL |
|---|---|---|
| `vision:{hash}` | Cached Ollama vision JSON responses | 24 hours |

Same sketch uploaded twice skips most Ollama calls on the second run.

### Files on disk

| Path | Contents |
|---|---|
| `backend/data/uploads/{job_id}.png` | Original upload (temporary) |
| `training_testing_datasets/feedback/images/` | Correction sketch copies |
| `training_testing_datasets/feedback/labels/` | Correction filled JSON files |

---

## Results UI

After `event: result`, [`MatchResults`](../frontend/components/MatchResults.tsx) displays:

1. **Header** — Master name, category, Encore `_id`, confidence badge
2. **Score breakdown** — Retrieval / Vision shape / Feedback boost / Combined
3. **Warnings** — Low vision score, validation issues
4. **Side-by-side images** — Your sketch (`/api/v1/match/{job_id}/upload`) vs master PNG
5. **Dimensions table** — Template lengths vs extracted lengths per segment
6. **JSON preview** — Full filled Encore output
7. **Download JSON** — Client-side blob download
8. **Correct this match** — Opens correction panel (see below)

---

## Correction / training flow

When the match is wrong:

1. Click **Correct this match**
2. Select correct master from dropdown (all 48)
3. Edit segment lengths
4. Click **Save correction & train**

**Backend (`POST /api/v1/feedback`):**
- Copies sketch to `feedback/images/`
- Writes filled JSON to `feedback/labels/` and Postgres `corrections`
- Appends to `feedback/manifest.jsonl`
- Updates retriever feedback entries in memory

**Future matches:** Similar sketches get retrieval boost (+50) and optional feedback image boost (+60) toward corrected masters. Masters previously corrected *away* from get −35 penalty.

This is **human-in-the-loop learning**, not model fine-tuning.

---

## Debugging with the live pipeline

### UI controls

| Control | Action |
|---|---|
| Expand step row | Show JSON debug payload for that step |
| Show raw JSON | Dump entire `liveTrace` array |
| Blue pulsing dot | Step currently running |
| Amber vision % | Vision shape score below 65% threshold |

### Key steps to inspect for wrong matches

1. **`analyze`** — Is `segment_count` correct? Is `part_class_hint` right?
2. **`retrieve`** — Is the correct master in top 5? What scores/reasons?
3. **`compare`** — What vision score did the wrong master get? Read `reasoning`.
4. **`match`** — Check `score_breakdown.vector_score`, `vision_score`, and `retrieval_score`.

### API health

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","postgres":true,"redis":true}
```

### Manual stream test

```bash
curl -N -X POST http://localhost:8000/api/v1/match/stream \
  -F "file=@training_testing_datasets/testing/Client_handwritten_data/your-sketch.png"
```

---

## Configuration reference

Environment variables in `backend/.env`:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_API_KEY` | — | Ollama Cloud authentication |
| `OLLAMA_VISION_MODEL` | `qwen3-vl:235b-instruct` | Sketch analysis + comparison |
| `OLLAMA_LLM_TEXT_MODEL` | `gpt-oss:120b-cloud` | Text reasoning (fallback selection) |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Text embeddings for pgvector retrieval |
| `RETRIEVAL_VECTOR_WEIGHT` | `0.35` | Weight of vector similarity in hybrid retrieve score |
| `RETRIEVAL_RULE_WEIGHT` | `0.65` | Weight of rule-based score in hybrid retrieve score |
| `VECTOR_SEARCH_TOP_K` | `20` | pgvector candidates fetched before hybrid merge |
| `DATABASE_URL` | `postgresql+asyncpg://encore:encore@localhost:5455/encore_drawings` | Postgres connection |
| `REDIS_URL` | `redis://localhost:6377/0` | Redis connection |
| `REDIS_CACHE_TTL_SECONDS` | `86400` | Vision + embed cache TTL |
| `MIN_VISION_SCORE` | `0.65` | Below this → warning + amber badge |
| `FEEDBACK_IMAGE_MATCH_THRESHOLD` | `0.72` | Min similarity for image boost |
| `FEEDBACK_IMAGE_BOOST` | `60` | Max boost from feedback image match |
| `WRONG_MASTER_PENALTY` | `35` | Penalty for previously wrong masters |

Frontend:

| Variable | Default |
|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` |

---

## Typical timing

| Phase | Duration (approx.) |
|---|---|
| Upload + save | < 1 s |
| Analyze (vision, cache miss) | 5–15 s |
| Retrieve (embed + pgvector + rules + feedback images) | 6–22 s |
| Compare × 5 candidates (vision) | 15–40 s |
| Extract lengths | 5–10 s |
| Validate + persist | < 1 s |
| **Total (cold, no cache)** | **30–90 s** |
| **Repeat same sketch (Redis hit)** | **Much faster** on cached vision calls |

---

## Related files

| Area | Path |
|---|---|
| Match API routes | [`backend/app/api/routes/match.py`](../backend/app/api/routes/match.py) |
| Match service | [`backend/app/services/match_service.py`](../backend/app/services/match_service.py) |
| Agent orchestrator | [`backend/app/features/agent/orchestrator.py`](../backend/app/features/agent/orchestrator.py) |
| RAG retriever | [`backend/app/features/rag/retriever.py`](../backend/app/features/rag/retriever.py) |
| pgvector search | [`backend/app/features/rag/vector_retriever.py`](../backend/app/features/rag/vector_retriever.py) |
| Embeddings | [`backend/app/features/embeddings/service.py`](../backend/app/features/embeddings/service.py) |
| Embed backfill script | [`backend/scripts/backfill_embeddings.py`](../backend/scripts/backfill_embeddings.py) |
| Vision analyzer | [`backend/app/features/vision/sketch_analyzer.py`](../backend/app/features/vision/sketch_analyzer.py) |
| Profile comparator | [`backend/app/features/vision/profile_comparator.py`](../backend/app/features/vision/profile_comparator.py) |
| Ollama + Redis cache | [`backend/app/features/ollama/client.py`](../backend/app/features/ollama/client.py) |
| Master catalog | [`backend/app/features/masters/catalog.py`](../backend/app/features/masters/catalog.py) |
| Frontend page | [`frontend/app/page.tsx`](../frontend/app/page.tsx) |
| SSE client | [`frontend/lib/api.ts`](../frontend/lib/api.ts) |
| Master dataset | [`training_testing_datasets/Training/Encore_master_drawings/`](../training_testing_datasets/Training/Encore_master_drawings/) |
