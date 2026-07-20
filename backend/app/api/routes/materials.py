from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.features.db.database_service import db_service
from app.features.db.models import Color, Material

router = APIRouter(prefix="/materials", tags=["materials"])


@router.get("")
async def list_materials():
    """All materials with their available colors nested."""
    if not db_service.enabled or db_service._factory is None:
        raise HTTPException(status_code=503, detail="Database not available")
    async with db_service._factory() as session:
        materials = (await session.scalars(select(Material).order_by(Material.name))).all()
        colors = (await session.scalars(select(Color).order_by(Color.name))).all()

    colors_by_material: dict[int, list[dict]] = {}
    for c in colors:
        colors_by_material.setdefault(c.material_id, []).append({"id": c.id, "name": c.name})

    return [
        {
            "id": m.id,
            "name": m.name,
            "density": m.density,
            "colors": colors_by_material.get(m.id, []),
        }
        for m in materials
    ]
