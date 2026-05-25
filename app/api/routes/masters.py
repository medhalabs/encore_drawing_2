from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.features.masters.catalog import MasterCatalog

router = APIRouter(prefix="/masters", tags=["masters"])


def get_catalog() -> MasterCatalog:
    from app.main import catalog
    return catalog


@router.get("")
def list_masters():
    return get_catalog().list_summaries()


@router.get("/{category}/{basename}")
def get_master(category: str, basename: str):
    key = f"{category}/{basename}"
    master = get_catalog().get_by_key(key)
    if not master:
        raise HTTPException(status_code=404, detail="Master not found")
    return master.drawing.to_encore_dict()


@router.get("/{category}/{basename}/image")
def get_master_image(category: str, basename: str):
    key = f"{category}/{basename}"
    master = get_catalog().get_by_key(key)
    if not master or not master.image_path.exists():
        raise HTTPException(status_code=404, detail="Master image not found")
    return FileResponse(master.image_path, media_type="image/png")
