import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import batch, feedback, health, masters, match, materials, training
from app.config.settings import get_settings
from app.features.agent.orchestrator import MatchOrchestrator
from app.features.classifier.efficientnet import EfficientNetClassifier
from app.features.classifier.retrain_service import RetrainService
from app.features.db.database_service import db_service
from app.features.embeddings.service import EmbeddingService
from app.features.feedback.store import FeedbackStore
from app.features.masters.catalog import MasterCatalog
from app.features.ollama.client import OllamaService
from app.features.rag.retriever import MasterRetriever
from app.features.vision.sketch_analyzer import SketchAnalyzer
from app.services.feedback_service import FeedbackService
from app.services.match_service import MatchService

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

catalog = MasterCatalog(settings)
feedback_store = FeedbackStore(settings, catalog)
ollama = OllamaService(settings)
analyzer = SketchAnalyzer(ollama, consensus_runs=1)
# retriever/embedding_service are kept for feedback-entry tracking and pgvector
# embedding backfill — the DL classifier alone picks the master (see MatchService)
retriever = MasterRetriever(catalog)
embedding_service = EmbeddingService(settings, ollama, catalog)
orchestrator = MatchOrchestrator(settings, analyzer)

# EfficientNet classifier — builds label index from all master keys (originals + mirrors)
_model_dir = Path(__file__).resolve().parents[1] / "data" / "models"
_label_index: list[str] = []  # populated after catalog.load()
classifier = EfficientNetClassifier(_model_dir, _label_index)
retrain_service = RetrainService(classifier, settings.master_drawings_dir, settings.feedback_path)

match_service = MatchService(settings, catalog, orchestrator, classifier=classifier)
feedback_service = FeedbackService(feedback_store, match_service)


async def _load_feedback_entries() -> list:
    feedback_store.load()
    if not db_service.enabled:
        return feedback_store.entries
    db_entries = await db_service.load_corrections()
    return db_entries if db_entries else feedback_store.entries


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    catalog.load()
    feedback_store.load()

    # Populate label index from catalog (must happen after catalog.load())
    all_keys = [m.key for m in catalog.masters]
    classifier._label_index = sorted(all_keys)
    classifier._key_to_idx = {k: i for i, k in enumerate(classifier._label_index)}

    # Load saved weights (uses label_index from training, may differ from current catalog order)
    classifier.load_if_ready()
    retrain_service.update_class_counts()

    # If no weights exist yet, kick off an initial training run in background
    if not list(_model_dir.glob("efficientnet_v*.pt")):
        retrain_service.retrain_now()

    try:
        await db_service.startup(settings, catalog, feedback_store.entries, embedding_service)
        print("PostgreSQL connected")
    except Exception as e:
        print(f"PostgreSQL unavailable: {e}")

    entries = await _load_feedback_entries()
    retriever.set_feedback_entries(entries)

    yield

    await db_service.shutdown()


app = FastAPI(title="Encore Drawing Matcher", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(masters.router, prefix="/api/v1")
app.include_router(match.router, prefix="/api/v1")
app.include_router(feedback.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")
app.include_router(batch.router, prefix="/api/v1")
app.include_router(materials.router, prefix="/api/v1")
