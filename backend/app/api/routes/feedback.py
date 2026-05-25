from fastapi import APIRouter, HTTPException

from app.core.models.schemas import FeedbackEntry, FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


def get_feedback_service():
    from app.main import feedback_service
    return feedback_service


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    try:
        return await get_feedback_service().submit(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("", response_model=list[FeedbackEntry])
async def list_feedback():
    return await get_feedback_service().list_entries()
