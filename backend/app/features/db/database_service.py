from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config.settings import Settings
from app.features.db import seed
from app.features.db.repository import DatabaseRepository
from app.features.db.session import create_tables, init_db
from app.features.embeddings.service import EmbeddingService
from app.features.masters.catalog import MasterCatalog
from app.features.rag.vector_retriever import VectorRetriever


class DatabaseService:
    def __init__(self):
        self._factory: async_sessionmaker[AsyncSession] | None = None
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def startup(
        self,
        settings: Settings,
        catalog: MasterCatalog,
        file_feedback_entries: list,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        if not settings.database_url:
            return
        try:
            init_db(settings)
            from app.features.db import session as db_session
            await create_tables()
            embedded = 0
            async with db_session._session_factory() as session:
                added = await seed.seed_masters_if_empty(session, catalog)
                imported = await seed.import_corrections_from_manifest(session, file_feedback_entries)
                if embedding_service:
                    try:
                        embedded = await seed.backfill_master_embeddings(
                            session, catalog, embedding_service
                        )
                    except Exception as e:
                        print(f"Embedding backfill skipped: {e}")
                        embedded = 0
            self._factory = db_session._session_factory
            self._enabled = True
            if added or imported or embedded:
                print(
                    f"DB seeded: {added} masters, {imported} corrections imported, "
                    f"{embedded} embeddings backfilled"
                )
        except Exception:
            from app.features.db.session import close_db
            await close_db()
            self._enabled = False
            self._factory = None
            raise

    async def shutdown(self) -> None:
        from app.features.db.session import close_db
        await close_db()
        self._enabled = False
        self._factory = None

    async def ping(self) -> bool:
        if not self._enabled or self._factory is None:
            return False
        try:
            async with self._factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    async def save_match(self, result, upload_path: str = "") -> None:
        if not self._enabled or self._factory is None:
            return
        async with self._factory() as session:
            await DatabaseRepository(session).save_match_job(result, upload_path)

    async def save_correction(self, entry, label_json: dict) -> None:
        if not self._enabled or self._factory is None:
            return
        async with self._factory() as session:
            await DatabaseRepository(session).save_correction(entry, label_json)

    async def load_corrections(self) -> list:
        if not self._enabled or self._factory is None:
            return []
        try:
            async with self._factory() as session:
                return await DatabaseRepository(session).list_corrections()
        except Exception:
            return []

    async def search_masters_by_embedding(
        self, query_vector: list[float], limit: int = 20
    ) -> dict[str, float]:
        if not self._enabled or self._factory is None:
            return {}
        async with self._factory() as session:
            return await VectorRetriever().search(session, query_vector, limit=limit)


db_service = DatabaseService()
