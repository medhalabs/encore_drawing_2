from sqlalchemy.ext.asyncio import AsyncSession

from app.features.db.repository import DatabaseRepository


class VectorRetriever:
    async def search(
        self, session: AsyncSession, query_vector: list[float], limit: int = 20
    ) -> dict[str, float]:
        hits = await DatabaseRepository(session).search_masters_by_embedding(
            query_vector, limit=limit
        )
        return {master_key: similarity for master_key, similarity in hits}
