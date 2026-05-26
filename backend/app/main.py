from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import feedback, health, masters, match
from app.config.settings import get_settings
from app.features.agent.orchestrator import MatchOrchestrator
from app.features.cache import redis_cache
from app.features.db.database_service import db_service
from app.features.embeddings.service import EmbeddingService
from app.features.feedback.store import FeedbackStore
from app.features.masters.catalog import MasterCatalog
from app.features.ollama.client import OllamaService
from app.features.rag.retriever import MasterRetriever
from app.features.vision.profile_comparator import ProfileComparator
from app.features.vision.sketch_analyzer import SketchAnalyzer
from app.services.feedback_service import FeedbackService
from app.services.match_service import MatchService

settings = get_settings()
catalog = MasterCatalog(settings)
feedback_store = FeedbackStore(settings, catalog)
ollama = OllamaService(settings)
analyzer = SketchAnalyzer(ollama)
retriever = MasterRetriever(catalog)
comparator = ProfileComparator(ollama)
embedding_service = EmbeddingService(settings, ollama, catalog)
orchestrator = MatchOrchestrator(
    settings, analyzer, retriever, comparator, ollama, feedback_store, embedding_service
)
match_service = MatchService(settings, catalog, orchestrator)
feedback_service = FeedbackService(feedback_store, match_service)


async def _load_feedback_entries() -> list:
    feedback_store.load()
    db_entries = await db_service.load_corrections()
    return db_entries if db_entries else feedback_store.entries


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    catalog.load()
    feedback_store.load()

    try:
        await redis_cache.init_redis(settings)
        print("Redis connected")
    except Exception as e:
        print(f"Redis unavailable: {e}")

    try:
        await db_service.startup(settings, catalog, feedback_store.entries, embedding_service)
        print("PostgreSQL connected")
    except Exception as e:
        print(f"PostgreSQL unavailable: {e}")

    entries = await _load_feedback_entries()
    retriever.set_feedback_entries(entries)

    yield

    await db_service.shutdown()
    await redis_cache.close_redis()


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
