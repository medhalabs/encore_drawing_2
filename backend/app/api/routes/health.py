from fastapi import APIRouter

from app.features.db.database_service import db_service

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "postgres": await db_service.ping(),
    }
