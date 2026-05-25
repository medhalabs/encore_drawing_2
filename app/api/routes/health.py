from fastapi import APIRouter

from app.features.cache import redis_cache
from app.features.db.database_service import db_service

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "postgres": await db_service.ping(),
        "redis": await redis_cache.ping_redis(),
    }
