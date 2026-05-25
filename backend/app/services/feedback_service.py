from app.core.models.schemas import FeedbackEntry, FeedbackRequest, FeedbackResponse
from app.features.db.database_service import db_service
from app.features.feedback.store import FeedbackStore
from app.services.match_service import MatchService


class FeedbackService:
    def __init__(self, store: FeedbackStore, match_service: MatchService):
        self.store = store
        self.match_service = match_service

    async def submit(self, request: FeedbackRequest) -> FeedbackResponse:
        previous_key = ""
        result = self.match_service.get_result(request.job_id)
        if result:
            previous_key = result.matched_master.key

        entry, filled_json = self.store.save_correction(request, previous_key)
        await db_service.save_correction(entry, filled_json)

        from app.main import retriever
        entries = await db_service.load_corrections()
        if entries:
            retriever.set_feedback_entries(entries)
        else:
            retriever.set_feedback_entries(self.store.entries)

        return FeedbackResponse(entry=entry, filled_json=filled_json)

    async def list_entries(self) -> list[FeedbackEntry]:
        entries = await db_service.load_corrections()
        return entries if entries else self.store.entries
