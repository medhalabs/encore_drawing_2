import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.config.settings import get_settings
from app.services.match_service import MatchService

router = APIRouter(prefix="/match", tags=["match"])


def get_match_service() -> MatchService:
    from app.main import match_service
    return match_service


@router.post("/stream")
async def match_drawing_stream(
    file: UploadFile = File(...),
    use_llm: bool = Form(True),
):
    settings = get_settings()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    service = get_match_service()
    sketch_path = service.save_upload(file.filename, content)

    async def event_generator():
        async for event in service.process_match_stream(sketch_path, file.filename, use_llm=use_llm):
            event_type = event["type"]
            payload = event["payload"]
            data = json.dumps(payload, default=str)
            yield f"event: {event_type}\ndata: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("")
async def match_drawing(
    file: UploadFile = File(...),
    use_llm: bool = Form(True),
):
    settings = get_settings()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    service = get_match_service()
    sketch_path = service.save_upload(file.filename, content)

    try:
        result = await service.process_match(sketch_path, file.filename, use_llm=use_llm)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{job_id}/upload")
def get_upload_image(job_id: str):
    settings = get_settings()
    upload_dir = settings.upload_path
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        path = upload_dir / f"{job_id}{ext}"
        if path.exists():
            media = "image/png" if ext == ".png" else "image/jpeg"
            return FileResponse(path, media_type=media)
    raise HTTPException(status_code=404, detail="Upload not found")


@router.get("/{job_id}/preprocessed")
def get_preprocessed_image(job_id: str):
    settings = get_settings()
    path = settings.upload_path / "preprocessed" / f"{job_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preprocessed image not found")
    return FileResponse(path, media_type="image/png")


@router.get("/{job_id}/export")
def export_json(job_id: str):
    service = get_match_service()
    result = service.get_result(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Match result not found")
    filename = f"{result.matched_master.key.replace('/', '-')}-filled.json"
    return JSONResponse(
        content=result.filled_json,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
