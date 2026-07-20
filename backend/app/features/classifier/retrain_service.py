"""
Watches the feedback store and triggers EfficientNet retraining every N new labels.
Also scans master drawings directory to seed training data.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from app.features.classifier.efficientnet import EfficientNetClassifier

logger = logging.getLogger(__name__)

RETRAIN_EVERY_N = 10  # trigger retrain after every N new corrections


class RetrainService:
    def __init__(
        self,
        classifier: EfficientNetClassifier,
        master_drawings_dir: Path,
        feedback_dir: Path,
    ):
        self.classifier = classifier
        self.master_drawings_dir = master_drawings_dir
        self.feedback_dir = feedback_dir
        # training_synth/<Category>/<stem>/ — same layout generate_synthetic_sketches.py
        # writes and scripts/train_from_dl.py reads. Without this the live incremental
        # retrain trains on ~50 master images instead of the ~14k synthetic corpus that
        # took accuracy from 24%->52%, and silently regresses the promoted model.
        self.synth_dir = master_drawings_dir.parent / "training_synth"
        # Persisted so a dev-server restart (--reload) doesn't reset this to 0 and fire
        # a retrain on the very next correction — that's what produced the regressed
        # efficientnet_v016.pt on 2026-07-16 (moved to data/models/discarded/).
        self._state_path = classifier.model_dir / "retrain_state.json"
        self._correction_count_at_last_retrain = self._load_state()

    def _load_state(self) -> int:
        try:
            return json.loads(self._state_path.read_text())["correction_count_at_last_retrain"]
        except Exception:
            return 0

    def _save_state(self) -> None:
        self._state_path.write_text(
            json.dumps({"correction_count_at_last_retrain": self._correction_count_at_last_retrain})
        )

    def seed_from_masters(self) -> list[tuple[Path, str]]:
        """Collect (image_path, master_key) pairs from the master catalog (base + mirror)."""
        items: list[tuple[Path, str]] = []
        for category_dir in sorted(self.master_drawings_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for png in sorted(category_dir.glob("*.png")):
                key = f"{category_dir.name}/{png.stem}"
                items.append((png, key))
        return items

    def seed_from_synthetic(self) -> list[tuple[Path, str]]:
        """Collect (image_path, master_key) pairs from the synthetic sketch corpus."""
        items: list[tuple[Path, str]] = []
        if not self.synth_dir.exists():
            return items
        for category_dir in sorted(self.synth_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for stem_dir in sorted(category_dir.iterdir()):
                if not stem_dir.is_dir():
                    continue
                key = f"{category_dir.name}/{stem_dir.name}"
                items += [(p, key) for p in sorted(stem_dir.glob("*.png"))]
        return items

    def collect_corrections(self) -> list[tuple[Path, str]]:
        """Collect labelled images from feedback manifest (ground-truth corrections)."""
        manifest = self.feedback_dir / "manifest.jsonl"
        if not manifest.exists():
            return []

        items: list[tuple[Path, str]] = []
        import json
        for line in manifest.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                image_rel = entry.get("image_path", "")
                master_key = entry.get("master_key", "")
                if not image_rel or not master_key:
                    continue
                image_path = self.feedback_dir / image_rel
                if image_path.exists():
                    items.append((image_path, master_key))
            except Exception:
                continue
        return items

    def build_training_set(self) -> list[tuple[Path, str]]:
        masters = self.seed_from_masters()
        synthetic = self.seed_from_synthetic()
        corrections = self.collect_corrections()
        return masters + synthetic + corrections

    def update_class_counts(self) -> None:
        items = self.build_training_set()
        counts = Counter(key for _, key in items)
        self.classifier.update_class_counts(dict(counts))

    def maybe_retrain(self, current_correction_count: int) -> bool:
        """
        Call this after every new correction is saved.
        Returns True if a retrain was triggered.
        """
        new_since_last = current_correction_count - self._correction_count_at_last_retrain
        if new_since_last < RETRAIN_EVERY_N:
            return False

        self._correction_count_at_last_retrain = current_correction_count
        self._save_state()
        training_data = self.build_training_set()
        self.update_class_counts()

        logger.info(
            "Triggering EfficientNet retrain — %d corrections, %d total training images",
            current_correction_count,
            len(training_data),
        )
        self.classifier.retrain_async(training_data)
        return True

    def retrain_now(self) -> None:
        """Force an immediate retrain (e.g. on startup when weights don't exist)."""
        training_data = self.build_training_set()
        self.update_class_counts()
        logger.info("Forcing retrain on %d training images", len(training_data))
        self.classifier.retrain_async(training_data)
