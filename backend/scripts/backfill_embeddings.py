#!/usr/bin/env -S uv run python
"""Re-embed all master drawings into pgvector."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings
from app.features.db import seed
from app.features.db.session import close_db, create_tables, init_db
from app.features.embeddings.service import EmbeddingService
from app.features.masters.catalog import MasterCatalog
from app.features.ollama.client import OllamaService


async def main() -> None:
    settings = get_settings()
    catalog = MasterCatalog(settings)
    catalog.load()
    ollama = OllamaService(settings)
    embedding_service = EmbeddingService(settings, ollama, catalog)

    init_db(settings)
    from app.features.db import session as db_session

    if db_session._session_factory is None:
        raise RuntimeError("Database not initialized")

    await create_tables()
    async with db_session._session_factory() as session:
        from sqlalchemy import select

        from app.features.db.models import MasterDrawing

        rows = (await session.scalars(select(MasterDrawing))).all()
        by_key = {m.key: m for m in catalog.masters}
        updated = 0
        for row in rows:
            master = by_key.get(row.master_key)
            if not master:
                continue
            row.embedding = embedding_service.embed_master(master)
            updated += 1
        await session.commit()
        print(f"Re-embedded {updated} master drawings")

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())
