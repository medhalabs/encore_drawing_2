import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.config.settings import Settings
from app.core.models.schemas import FeedbackEntry, FeedbackRequest
from app.features.masters.catalog import MasterCatalog
from app.features.matching.json_filler import fill_master_json


class FeedbackStore:
    def __init__(self, settings: Settings, catalog: MasterCatalog):
        self.settings = settings
        self.catalog = catalog
        self.root = settings.feedback_path
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.manifest_path = self.root / "manifest.jsonl"
        self._entries: list[FeedbackEntry] = []

    def load(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self._entries = []
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._entries.append(FeedbackEntry.model_validate(json.loads(line)))

    @property
    def entries(self) -> list[FeedbackEntry]:
        return self._entries

    def save_correction(
        self,
        request: FeedbackRequest,
        previous_master_key: str = "",
    ) -> tuple[FeedbackEntry, dict]:
        master = self.catalog.get_by_key(request.master_key)
        if not master:
            raise ValueError(f"Unknown master key: {request.master_key}")

        feedback_id = str(uuid.uuid4())
        upload_src = self.settings.upload_path
        upload_file = None
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            candidate = upload_src / f"{request.job_id}{ext}"
            if candidate.exists():
                upload_file = candidate
                break
        if not upload_file:
            raise ValueError("Original upload image not found for this job")

        image_dest = self.images_dir / f"{feedback_id}{upload_file.suffix}"
        shutil.copy2(upload_file, image_dest)

        filled = fill_master_json(master, request.lengths)
        filled_dict = filled.to_encore_dict()
        label_path = self.labels_dir / f"{feedback_id}.json"
        label_path.write_text(json.dumps(filled_dict, indent=2), encoding="utf-8")

        entry = FeedbackEntry(
            feedback_id=feedback_id,
            job_id=request.job_id,
            master_key=request.master_key,
            master_id=master.drawing.id,
            segment_count=len(request.lengths),
            angles=master.drawing.angles,
            part_class=master.drawing.part_class,
            lengths=request.lengths,
            note=request.note or "",
            image_path=str(image_dest.relative_to(self.root)),
            label_path=str(label_path.relative_to(self.root)),
            created_at=datetime.now(timezone.utc).isoformat(),
            previous_master_key=previous_master_key,
        )

        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

        self._entries.append(entry)
        return entry, filled_dict
