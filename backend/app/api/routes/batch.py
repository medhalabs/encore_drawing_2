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

from app.config.settings import get_settings

router = APIRouter(prefix="/batch", tags=["batch"])
logger = logging.getLogger(__name__)


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
    crop_path = settings.upload_path / f"batch_{batch_id}_crops" / f"crop_{index:02d}.png"
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Crop not found")
    return FileResponse(str(crop_path), media_type="image/png")
