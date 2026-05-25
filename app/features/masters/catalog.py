from app.core.models.schemas import MasterSummary
from app.features.masters.loader import MasterRecord, load_all_masters
from app.config.settings import Settings


class MasterCatalog:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._masters: list[MasterRecord] = []
        self._by_key: dict[str, MasterRecord] = {}
        self._by_id: dict[str, MasterRecord] = {}

    def load(self) -> None:
        self._masters = load_all_masters(self.settings.master_drawings_dir)
        self._by_key = {m.key: m for m in self._masters}
        self._by_id = {m.drawing.id: m for m in self._masters}

    @property
    def masters(self) -> list[MasterRecord]:
        return self._masters

    def get_by_key(self, key: str) -> MasterRecord | None:
        return self._by_key.get(key)

    def get_by_id(self, master_id: str) -> MasterRecord | None:
        return self._by_id.get(master_id)

    def list_summaries(self) -> list[MasterSummary]:
        return [
            MasterSummary(
                key=m.key,
                id=m.drawing.id,
                name=m.display_name,
                category=m.category,
                segment_count=m.segment_count,
                part_class=m.drawing.part_class,
                image_url=f"/api/v1/masters/{m.key}/image",
            )
            for m in self._masters
        ]

    def fingerprint(self, master: MasterRecord) -> str:
        d = master.drawing
        folds = []
        if d.start_fold_type:
            folds.append(f"start_fold={d.start_fold_type}:{d.start_fold_length}")
        if d.end_fold_type:
            folds.append(f"end_fold={d.end_fold_type}:{d.end_fold_length}")
        return (
            f"{master.key} | name={d.name} | partClass={d.part_class} | "
            f"segments={len(d.lengths)} | angles={d.angles} | direction={d.direction} | "
            f"firstSegmentAngle={d.first_segment_angle} | folds={','.join(folds) or 'none'}"
        )
