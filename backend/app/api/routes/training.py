"""
Training management routes.

POST /training/upload   — upload labelled sketch images for a master key
POST /training/retrain  — force an immediate retrain
GET  /training/status   — class image counts, model version, training state
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/training", tags=["training"])

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}


def _get_deps():
    from app.main import catalog, classifier, retrain_service, settings
    return catalog, classifier, retrain_service, settings


# ── Schemas ────────────────────────────────────────────────────────────────────

class UploadResult(BaseModel):
    saved: int
    skipped: int
    master_key: str
    filenames: list[str]


class RetrainResponse(BaseModel):
    triggered: bool
    message: str
    total_training_images: int


class ClassStat(BaseModel):
    key: str
    category: str
    name: str
    master_count: int      # original master PNGs
    correction_count: int  # labelled hand-drawn sketches
    total: int


class TrainingStatus(BaseModel):
    is_training: bool
    model_version: str | None
    total_classes: int
    total_training_images: int
    classes: list[ClassStat]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _feedback_counts(feedback_dir: Path) -> dict[str, int]:
    """Count labelled corrections per master_key from manifest."""
    manifest = feedback_dir / "manifest.jsonl"
    counts: dict[str, int] = {}
    if not manifest.exists():
        return counts
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            key = entry.get("master_key", "")
            img_path = entry.get("image_path", "")
            if key and img_path and (feedback_dir / img_path).exists():
                counts[key] = counts.get(key, 0) + 1
        except Exception:
            continue
    return counts


def _latest_model_version(model_dir: Path) -> str | None:
    pts = sorted(model_dir.glob("efficientnet_v*.pt"))
    return pts[-1].stem if pts else None


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResult)
async def upload_training_images(
    master_key: str = Form(..., description="e.g. Aprons/apron-1"),
    files: list[UploadFile] = File(...),
):
    """Upload one or more hand-drawn sketches labelled for a specific master."""
    catalog, _, retrain_service, settings = _get_deps()

    master = catalog.get_by_key(master_key)
    if not master:
        raise HTTPException(status_code=404, detail=f"Unknown master key: {master_key}")

    feedback_dir: Path = settings.feedback_path
    images_dir = feedback_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = feedback_dir / "manifest.jsonl"

    saved = 0
    skipped = 0
    filenames: list[str] = []

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for upload in files:
            if upload.content_type not in ALLOWED_TYPES:
                skipped += 1
                continue

            content = await upload.read()
            if not content:
                skipped += 1
                continue

            ext = Path(upload.filename or "sketch.png").suffix or ".png"
            file_id = str(uuid.uuid4())
            dest = images_dir / f"{file_id}{ext}"
            dest.write_bytes(content)

            entry = {
                "feedback_id": file_id,
                "job_id": file_id,
                "master_key": master_key,
                "master_id": master.drawing.id,
                "segment_count": master.segment_count,
                "angles": master.drawing.angles,
                "part_class": master.drawing.part_class,
                "lengths": [],
                "note": "training_upload",
                "image_path": str(dest.relative_to(feedback_dir)),
                "label_path": "",
                "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "previous_master_key": "",
            }
            manifest.write(json.dumps(entry) + "\n")
            saved += 1
            filenames.append(upload.filename or dest.name)

    # Update classifier class counts so status reflects new images
    retrain_service.update_class_counts()

    return UploadResult(saved=saved, skipped=skipped, master_key=master_key, filenames=filenames)


@router.post("/retrain", response_model=RetrainResponse)
def trigger_retrain():
    """Force an immediate EfficientNet retrain on all masters + corrections."""
    _, classifier, retrain_service, _ = _get_deps()

    if classifier._training:
        return RetrainResponse(
            triggered=False,
            message="Retrain already in progress — wait for it to finish.",
            total_training_images=len(retrain_service.build_training_set()),
        )

    training_data = retrain_service.build_training_set()
    retrain_service.retrain_now()

    return RetrainResponse(
        triggered=True,
        message=f"Retrain started in background on {len(training_data)} images (10 epochs). Hot-swaps when done.",
        total_training_images=len(training_data),
    )


class TrainingImage(BaseModel):
    feedback_id: str
    master_key: str
    filename: str
    created_at: str
    image_url: str


@router.get("/images")
def list_training_images(master_key: str):
    """Return all uploaded training images for a given master key."""
    _, _, _, settings = _get_deps()
    manifest = settings.feedback_path / "manifest.jsonl"
    results: list[dict] = []
    if not manifest.exists():
        return results
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("master_key") != master_key:
            continue
        fid = entry.get("feedback_id", "")
        img_path = entry.get("image_path", "")
        full_path = settings.feedback_path / img_path if img_path else None
        if not full_path or not full_path.exists():
            continue
        results.append({
            "feedback_id": fid,
            "master_key": master_key,
            "filename": Path(img_path).name,
            "created_at": entry.get("created_at", ""),
            "image_url": f"/api/v1/training/images/{fid}/file",
        })
    return results


@router.get("/images/{feedback_id}/file")
def get_training_image_file(feedback_id: str):
    """Serve the raw image file for a training image."""
    _, _, _, settings = _get_deps()
    manifest = settings.feedback_path / "manifest.jsonl"
    if not manifest.exists():
        raise HTTPException(status_code=404, detail="Not found")
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("feedback_id") != feedback_id:
            continue
        img_path = entry.get("image_path", "")
        full_path = settings.feedback_path / img_path if img_path else None
        if full_path and full_path.exists():
            media = "image/png" if full_path.suffix == ".png" else "image/jpeg"
            return FileResponse(str(full_path), media_type=media)
    raise HTTPException(status_code=404, detail="Training image not found")


@router.delete("/images/{feedback_id}")
def delete_training_image(feedback_id: str):
    """Remove a training image from the manifest and disk."""
    _, _, retrain_service, settings = _get_deps()
    manifest_path = settings.feedback_path / "manifest.jsonl"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Not found")

    lines = manifest_path.read_text().splitlines()
    kept: list[str] = []
    deleted_entry: dict | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except Exception:
            kept.append(line)
            continue
        if entry.get("feedback_id") == feedback_id:
            deleted_entry = entry
        else:
            kept.append(line)

    if deleted_entry is None:
        raise HTTPException(status_code=404, detail="Training image not found")

    # Rewrite manifest atomically
    tmp = manifest_path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    tmp.replace(manifest_path)

    # Delete file from disk
    img_path = deleted_entry.get("image_path", "")
    if img_path:
        full_path = settings.feedback_path / img_path
        if full_path.exists():
            full_path.unlink()

    # Refresh class counts
    retrain_service.update_class_counts()
    return {"deleted": feedback_id}


@router.post("/stop")
def stop_training():
    """Signal the running retrain to stop after the current epoch."""
    _, classifier, _, _ = _get_deps()
    was_training = classifier.stop_training()
    if was_training:
        return {"stopped": True, "message": "Stop signal sent — training will halt after the current epoch."}
    return {"stopped": False, "message": "No training is currently running."}


@router.post("/restart", response_model=RetrainResponse)
def restart_training():
    """Stop any running retrain and immediately start a fresh one."""
    _, classifier, retrain_service, _ = _get_deps()

    # Signal stop if running (new retrain will start right after)
    classifier.stop_training()

    # Small wait to let the stop propagate before the new thread starts
    import time
    for _ in range(20):
        if not classifier._training:
            break
        time.sleep(0.1)

    training_data = retrain_service.build_training_set()
    retrain_service.update_class_counts()
    classifier._stop_event.clear()
    classifier.retrain_async(training_data)

    return RetrainResponse(
        triggered=True,
        message=f"Restarted training on {len(training_data)} images (10 epochs).",
        total_training_images=len(training_data),
    )


@router.get("/progress")
def training_progress():
    """Return live epoch-by-epoch training progress."""
    _, classifier, _, _ = _get_deps()
    current = classifier._epoch_current
    total = classifier._epoch_total
    losses = list(classifier._epoch_losses)
    is_training = classifier._training
    images_done = classifier._images_processed
    images_per_epoch = classifier._images_per_epoch
    total_images = classifier._training_images_count
    percent = round((current / total) * 100) if total > 0 and is_training else (100 if not is_training and losses else 0)
    return {
        "is_training": is_training,
        "current_epoch": current,
        "total_epochs": total,
        "epoch_losses": losses,
        "current_loss": losses[-1] if losses else None,
        "percent": percent,
        "total_images": total_images,
        "images_processed": images_done,
        "images_per_epoch": images_per_epoch,
    }


@router.get("/status", response_model=TrainingStatus)
def training_status():
    """Return per-class image counts, model version, and training state."""
    catalog, classifier, retrain_service, settings = _get_deps()

    model_dir = settings.upload_path.parent / "models"
    correction_counts = _feedback_counts(settings.feedback_path)

    # Count master originals per key (excluding mirrors)
    master_counts: dict[str, int] = {}
    for m in catalog.masters:
        if not m.key.endswith("-mirror"):
            master_counts[m.key] = 1

    classes: list[ClassStat] = []
    for m in catalog.masters:
        if m.key.endswith("-mirror"):
            continue
        corrections = correction_counts.get(m.key, 0)
        master_img = master_counts.get(m.key, 0)
        classes.append(ClassStat(
            key=m.key,
            category=m.category,
            name=m.display_name,
            master_count=master_img,
            correction_count=corrections,
            total=master_img + corrections,
        ))

    classes.sort(key=lambda c: c.key)
    total_images = sum(c.total for c in classes)

    return TrainingStatus(
        is_training=classifier._training,
        model_version=_latest_model_version(model_dir),
        total_classes=len(classes),
        total_training_images=total_images,
        classes=classes,
    )
