import json
from dataclasses import dataclass
from pathlib import Path

from app.core.models.schemas import EncoreDrawing


@dataclass
class MasterRecord:
    key: str
    category: str
    basename: str
    json_path: Path
    image_path: Path
    drawing: EncoreDrawing

    @property
    def segment_count(self) -> int:
        return len(self.drawing.lengths)

    @property
    def display_name(self) -> str:
        if self.drawing.name.strip():
            return self.drawing.name.strip()
        return self.basename


def load_master_record(category: str, json_path: Path) -> MasterRecord | None:
    basename = json_path.stem
    image_path = json_path.with_suffix(".png")
    if not image_path.exists():
        return None

    with json_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    drawing = EncoreDrawing.model_validate(raw)
    key = f"{category}/{basename}"

    return MasterRecord(
        key=key,
        category=category,
        basename=basename,
        json_path=json_path,
        image_path=image_path,
        drawing=drawing,
    )


def load_all_masters(root_dir: Path) -> list[MasterRecord]:
    masters: list[MasterRecord] = []
    if not root_dir.exists():
        return masters

    for category_dir in sorted(root_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for json_path in sorted(category_dir.glob("*.json")):
            record = load_master_record(category_dir.name, json_path)
            if record:
                masters.append(record)

    return masters
