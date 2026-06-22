"""
EfficientNet-B0 classifier for matching handwritten sketches to master drawings.

Fast path: if confidence >= threshold, skip LLM comparison entirely.
Retrains automatically every N new labelled images (background thread).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

logger = logging.getLogger(__name__)

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),   # sketches are greyscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_RETRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.0),        # do NOT flip — that changes meaning
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MIN_IMAGES_PER_CLASS = 10  # Below this, classifier is not trusted
HIGH_CONFIDENCE_THRESHOLD = 0.85  # Skip LLM if confidence >= this


@dataclass
class ClassifierResult:
    master_key: str
    confidence: float
    is_mirror: bool


class EfficientNetClassifier:
    """
    Wraps EfficientNet-B0 with:
    - label_index: sorted list of master keys (class names)
    - Hot-swap weights without server restart
    - Thread-safe predict / retrain
    """

    def __init__(self, model_dir: Path, label_index: list[str]):
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._label_index = label_index          # sorted list of master_key strings
        self._key_to_idx = {k: i for i, k in enumerate(label_index)}
        self._lock = threading.RLock()
        self._model: nn.Module | None = None
        self._training = False

        # Track how many images each class has — only trust classifier when enough data
        self._class_counts: dict[str, int] = {}

        # Weights are loaded lazily via load_if_ready() after the label index is populated
        logger.info("EfficientNet classifier initialised — call load_if_ready() after setting label_index")

    # ── Public API ────────────────────────────────────────────────────

    def predict(self, image_path: Path) -> ClassifierResult | None:
        """
        Returns a ClassifierResult if the model is loaded and confident.
        Returns None if not enough data or model not loaded — caller falls back to LLM.
        """
        with self._lock:
            if self._model is None:
                return None

        img = Image.open(image_path).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0)

        with self._lock:
            self._model.eval()
            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                conf, idx = probs.max(0)
                confidence = float(conf)
                class_idx = int(idx)

        if class_idx >= len(self._label_index):
            return None

        key = self._label_index[class_idx]

        # Don't trust the classifier for classes with very little data
        # Masters always have at least 1 image; require more for correction-only classes
        base_key = key.replace("-mirror", "")
        count = self._class_counts.get(base_key, 0)
        if count < 1:
            return None

        is_mirror = key.endswith("-mirror")
        return ClassifierResult(master_key=key, confidence=confidence, is_mirror=is_mirror)

    def is_confident(self, result: ClassifierResult | None) -> bool:
        return result is not None and result.confidence >= HIGH_CONFIDENCE_THRESHOLD

    def load_if_ready(self) -> None:
        """Load saved weights if available — call after label_index is populated."""
        index_path = self.model_dir / "label_index.json"
        weights_path = self._latest_weights()
        if weights_path and weights_path.exists() and index_path.exists():
            saved_index = json.loads(index_path.read_text())
            # Use saved index (it was the one used during training)
            self._label_index = saved_index
            self._key_to_idx = {k: i for i, k in enumerate(self._label_index)}
            self._load_weights(weights_path)
            logger.info("Classifier loaded from %s (%d classes)", weights_path.name, len(self._label_index))
        else:
            logger.info("No classifier weights found — cold start, LLM will be used until retrain")

    def update_class_counts(self, counts: dict[str, int]) -> None:
        self._class_counts = counts

    def retrain_async(
        self,
        training_images: list[tuple[Path, str]],  # (image_path, master_key)
        on_complete: callable | None = None,
    ) -> None:
        """Kick off background retraining without blocking the request thread."""
        if self._training:
            logger.info("Retrain already in progress, skipping")
            return
        t = threading.Thread(target=self._retrain, args=(training_images, on_complete), daemon=True)
        t.start()

    # ── Internal ──────────────────────────────────────────────────────

    def _latest_weights(self) -> Path | None:
        pts = sorted(self.model_dir.glob("efficientnet_v*.pt"))
        return pts[-1] if pts else None

    def _load_weights(self, path: Path) -> None:
        n_classes = len(self._label_index)
        model = _build_model(n_classes)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with self._lock:
            self._model = model

    def _retrain(
        self,
        training_images: list[tuple[Path, str]],
        on_complete: callable | None,
    ) -> None:
        self._training = True
        try:
            logger.info("Retraining EfficientNet on %d images …", len(training_images))

            # Build label index from ALL known master keys (preserve order)
            keys_in_data = sorted({k for _, k in training_images})
            for k in keys_in_data:
                if k not in self._key_to_idx:
                    self._label_index.append(k)
                    self._key_to_idx[k] = len(self._label_index) - 1

            n_classes = len(self._label_index)
            model = _build_model(n_classes)

            # Load existing weights if available (fine-tune, not from scratch)
            existing = self._latest_weights()
            if existing:
                try:
                    state = torch.load(existing, map_location="cpu", weights_only=True)
                    # If n_classes changed, only load compatible layers
                    if state["classifier.1.weight"].shape[0] == n_classes:
                        model.load_state_dict(state)
                        logger.info("Fine-tuning from %s", existing.name)
                except Exception:
                    logger.warning("Couldn't load previous weights, starting fresh")

            dataset = _SketchDataset(training_images, self._key_to_idx, _RETRAIN_TRANSFORM)
            loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for epoch in range(10):
                total_loss = 0.0
                for images, labels in loader:
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                logger.info("Epoch %d/%d  loss=%.4f", epoch + 1, 10, total_loss / max(len(loader), 1))

            # Save versioned weights
            existing_pts = sorted(self.model_dir.glob("efficientnet_v*.pt"))
            next_v = len(existing_pts) + 1
            out_path = self.model_dir / f"efficientnet_v{next_v:03d}.pt"
            torch.save(model.state_dict(), out_path)

            # Save label index alongside weights
            index_path = self.model_dir / "label_index.json"
            index_path.write_text(json.dumps(self._label_index))

            # Hot-swap — live requests will use new model immediately
            model.eval()
            with self._lock:
                self._model = model

            logger.info("Retrain complete — saved %s (%d classes)", out_path.name, n_classes)
            if on_complete:
                on_complete(out_path)

        except Exception:
            logger.exception("Retrain failed")
        finally:
            self._training = False


def _build_model(n_classes: int) -> nn.Module:
    import ssl
    import torch.hub as hub

    # macOS Python 3.13 ships without root certs — bypass SSL for the one-time weights download
    _orig_ssl = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    _orig_env = hub._get_torch_home  # noqa: SLF001 — we restore immediately after

    try:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    finally:
        ssl._create_default_https_context = _orig_ssl

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)
    return model


class _SketchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        items: list[tuple[Path, str]],
        key_to_idx: dict[str, int],
        transform,
    ):
        self.items = [(p, key_to_idx[k]) for p, k in items if k in key_to_idx]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
