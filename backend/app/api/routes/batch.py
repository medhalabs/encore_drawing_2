"""
Batch PDF / multi-drawing upload endpoint.

POST /batch/upload
  - Accepts a PDF or image file
  - Splits into individual drawing crops
  - Runs classifier on each crop
  - Returns list of job results (one per drawing)

GET /batch/{batch_id}/crops/{index}
  - Returns the cropped image for a specific drawing
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from app.config.settings import get_settings

router = APIRouter(prefix="/batch", tags=["batch"])
logger = logging.getLogger(__name__)


# ── HITL box-review schemas ────────────────────────────────────────────────
class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int


class PageSelection(BaseModel):
    index: int          # page index
    boxes: list[Box]


class ClassifyRequest(BaseModel):
    use_llm: bool = True
    pages: list[PageSelection]


def _get_match_service():
    from app.main import match_service
    return match_service


@router.post("/upload")
async def batch_upload(
    file: UploadFile = File(...),
    use_llm: bool = Form(True),
):
    """
    Upload a PDF or image containing multiple drawings.
    Returns a batch_id and list of per-drawing results streamed as SSE.
    """
    from app.features.vision.batch_splitter import split_file

    settings = get_settings()
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit for PDFs
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    # Save the upload
    batch_id = uuid.uuid4().hex
    suffix = Path(file.filename or "upload.pdf").suffix.lower()
    upload_path = settings.upload_path / f"batch_{batch_id}{suffix}"
    upload_path.write_bytes(content)

    # Split into individual drawings
    crops = split_file(upload_path)
    if not crops:
        raise HTTPException(status_code=422, detail="No drawings detected in file")

    logger.info(
        "batch_upload id=%s file=%s crops=%d use_llm=%s",
        batch_id,
        file.filename,
        len(crops),
        use_llm,
    )

    # Save each crop
    crops_dir = settings.upload_path / f"batch_{batch_id}_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = []
    for i, crop_img in enumerate(crops):
        crop_path = crops_dir / f"crop_{i:02d}.png"
        crop_img.save(crop_path)
        crop_paths.append(crop_path)

    service = _get_match_service()

    async def event_generator():
        # First emit the total count so frontend knows how many to expect
        yield f"event: start\ndata: {json.dumps({'batch_id': batch_id, 'total': len(crop_paths)})}\n\n"

        for i, crop_path in enumerate(crop_paths):
            logger.info(
                "batch %s processing crop %d/%d (%s)",
                batch_id,
                i + 1,
                len(crop_paths),
                crop_path.name,
            )
            # Stream match result for this crop
            result_payload = None
            async for event in service.process_match_stream(
                crop_path,
                crop_path.name,
                use_llm=use_llm,
            ):
                if event["type"] == "result":
                    result_payload = event["payload"]

            job_id = result_payload.get("job_id", "") if result_payload else ""
            logger.info(
                "batch %s crop %d/%d done job_id=%s matched=%s",
                batch_id,
                i + 1,
                len(crop_paths),
                job_id,
                result_payload.get("matched_master", {}).get("key") if result_payload and result_payload.get("matched_master") else None,
            )
            payload = {
                "index": i,
                "job_id": job_id,
                "batch_id": batch_id,
                "crop_url": f"/api/v1/batch/{batch_id}/crops/{i}",
                "result": result_payload,
            }
            yield f"event: drawing\ndata: {json.dumps(payload, default=str)}\n\n"

        yield f"event: done\ndata: {json.dumps({'batch_id': batch_id})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{batch_id}/crops/{index}")
def get_crop(batch_id: str, index: int):
    """Serve a specific crop image."""
    settings = get_settings()
    # HITL classify flow saves crops into upload_path; the legacy /upload flow
    # used a per-batch crops dir — check both.
    candidates = [
        settings.upload_path / f"batchcrop_{batch_id}_{index:02d}.png",
        settings.upload_path / f"batch_{batch_id}_crops" / f"crop_{index:02d}.png",
    ]
    for crop_path in candidates:
        if crop_path.exists():
            return FileResponse(str(crop_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="Crop not found")


# ── Human-in-the-loop flow: detect → review/adjust boxes → classify ────────

@router.post("/detect")
async def batch_detect(file: UploadFile = File(...)):
    """
    Stage 1 of the HITL flow. Render the page(s), auto-detect a bounding box
    around each drawing, and return the page images + boxes for the user to
    review/adjust. No classification happens here.
    """
    from app.features.vision.box_detector import detect_file

    settings = get_settings()
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    batch_id = uuid.uuid4().hex
    suffix = Path(file.filename or "upload.pdf").suffix.lower()
    upload_path = settings.upload_path / f"batch_{batch_id}{suffix}"
    upload_path.write_bytes(content)

    pages = detect_file(upload_path)
    if not pages:
        raise HTTPException(status_code=422, detail="Could not read any pages from file")

    pages_dir = settings.upload_path / f"batch_{batch_id}_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    payload_pages = []
    for pb in pages:
        pb.image.save(pages_dir / f"page_{pb.index:02d}.png")
        payload_pages.append({
            "index": pb.index,
            "image_url": f"/api/v1/batch/{batch_id}/page/{pb.index}",
            "width": pb.width,
            "height": pb.height,
            "boxes": pb.boxes,
        })

    logger.info("batch_detect id=%s file=%s pages=%d boxes=%d",
                batch_id, file.filename, len(pages), sum(len(p["boxes"]) for p in payload_pages))
    return {"batch_id": batch_id, "pages": payload_pages}


@router.get("/{batch_id}/page/{index}")
def get_page(batch_id: str, index: int):
    """Serve a rendered page image for the box-review UI."""
    settings = get_settings()
    page_path = settings.upload_path / f"batch_{batch_id}_pages" / f"page_{index:02d}.png"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(str(page_path), media_type="image/png")


@router.post("/{batch_id}/classify")
async def batch_classify(batch_id: str, req: ClassifyRequest):
    """
    Stage 2 of the HITL flow. Crop the user-confirmed boxes out of the rendered
    pages, then classify each crop, streaming results as SSE (same shape as
    /batch/upload so the results UI is shared).
    """
    from PIL import Image

    settings = get_settings()
    pages_dir = settings.upload_path / f"batch_{batch_id}_pages"
    if not pages_dir.exists():
        raise HTTPException(status_code=404, detail="Unknown batch_id — run /detect first")

    # Crop every confirmed box, in page then top-left order.
    # Save into upload_path with a job-id-friendly name so /feedback (which
    # resolves the source image as upload_path/{job_id}.png) works for crops.
    crop_paths: list[Path] = []
    for page in req.pages:
        page_path = pages_dir / f"page_{page.index:02d}.png"
        if not page_path.exists():
            continue
        page_img = Image.open(page_path).convert("RGB")
        pw, ph = page_img.size
        for box in page.boxes:
            x1 = max(0, min(box.x, pw - 1))
            y1 = max(0, min(box.y, ph - 1))
            x2 = max(x1 + 1, min(box.x + box.w, pw))
            y2 = max(y1 + 1, min(box.y + box.h, ph))
            crop = page_img.crop((x1, y1, x2, y2))
            crop_path = settings.upload_path / f"batchcrop_{batch_id}_{len(crop_paths):02d}.png"
            crop.save(crop_path)
            crop_paths.append(crop_path)

    if not crop_paths:
        raise HTTPException(status_code=422, detail="No boxes provided")

    service = _get_match_service()

    async def event_generator():
        yield f"event: start\ndata: {json.dumps({'batch_id': batch_id, 'total': len(crop_paths)})}\n\n"
        for i, crop_path in enumerate(crop_paths):
            result_payload = None
            async for event in service.process_match_stream(
                crop_path, crop_path.name, use_llm=req.use_llm,
            ):
                if event["type"] == "result":
                    result_payload = event["payload"]
            job_id = result_payload.get("job_id", "") if result_payload else ""
            payload = {
                "index": i,
                "job_id": job_id,
                "batch_id": batch_id,
                "crop_url": f"/api/v1/batch/{batch_id}/crops/{i}",
                "result": result_payload,
            }
            yield f"event: drawing\ndata: {json.dumps(payload, default=str)}\n\n"
        yield f"event: done\ndata: {json.dumps({'batch_id': batch_id})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
