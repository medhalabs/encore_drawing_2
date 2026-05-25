from app.core.models.schemas import EncoreDrawing
from app.features.masters.loader import MasterRecord


def fill_master_json(master: MasterRecord, extracted_lengths: list[float]) -> EncoreDrawing:
    data = master.drawing.model_dump(by_alias=True)
    data["lengths"] = extracted_lengths
    if not data.get("name"):
        data["name"] = master.display_name
    return EncoreDrawing.model_validate(data)
