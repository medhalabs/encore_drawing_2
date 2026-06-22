"""
Watches the feedback store and triggers EfficientNet retraining every N new labels.
Also scans master drawings directory to seed training data.
"""

from __future__ import annotations

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
        self._correction_count_at_last_retrain = 0

    def seed_from_masters(self) -> list[tuple[Path, str]]:
        """Collect (image_path, master_key) pairs from the master catalog (originals only)."""
        items: list[tuple[Path, str]] = []
        for category_dir in sorted(self.master_drawings_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for png in sorted(category_dir.glob("*.png")):
                if "-mirror" in png.stem:
                    continue
                key = f"{category_dir.name}/{png.stem}"
                items.append((png, key))
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
        corrections = self.collect_corrections()
        # Corrections override / augment master seeds
        return masters + corrections

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
