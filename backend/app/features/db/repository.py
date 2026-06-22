from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.models.schemas import FeedbackEntry, MatchResult
from app.features.db.models import Correction, MasterDrawing, MatchJob


class DatabaseRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_match_job(self, result: MatchResult, upload_path: str = "") -> None:
        existing = await self.session.get(MatchJob, result.job_id)
        payload = {
            "upload_path": upload_path,
            "matched_master_key": result.matched_master.key if result.matched_master else "",
            "matched_encore_id": result.matched_master.id if result.matched_master else "",
            "confidence": result.confidence,
            "extracted_lengths": result.extracted_lengths,
            "filled_json": result.filled_json,
            "score_breakdown": result.score_breakdown.model_dump() if result.score_breakdown else None,
            "agent_trace": [s.model_dump() for s in result.agent_trace],
            "warnings": result.warnings,
        }
        if existing:
            for k, v in payload.items():
                setattr(existing, k, v)
        else:
            self.session.add(MatchJob(job_id=result.job_id, **payload))
        await self.session.commit()

    async def get_match_job(self, job_id: str) -> MatchJob | None:
        return await self.session.get(MatchJob, job_id)

    async def save_correction(self, entry: FeedbackEntry, label_json: dict) -> None:
        self.session.add(
            Correction(
                feedback_id=entry.feedback_id,
                job_id=entry.job_id,
                master_key=entry.master_key,
                master_id=entry.master_id,
                segment_count=entry.segment_count,
                angles=entry.angles,
                part_class=entry.part_class,
                lengths=entry.lengths,
                note=entry.note,
                image_path=entry.image_path,
                label_json=label_json,
                previous_master_key=entry.previous_master_key,
            )
        )
        await self.session.commit()

    async def list_corrections(self) -> list[FeedbackEntry]:
        rows = (await self.session.scalars(select(Correction).order_by(Correction.created_at))).all()
        return [
            FeedbackEntry(
                feedback_id=r.feedback_id,
                job_id=r.job_id,
                master_key=r.master_key,
                master_id=r.master_id,
                segment_count=r.segment_count,
                angles=r.angles,
                part_class=r.part_class,
                lengths=r.lengths,
                note=r.note or "",
                image_path=r.image_path,
                label_path="",
                created_at=r.created_at.isoformat(),
                previous_master_key=r.previous_master_key or "",
            )
            for r in rows
        ]

    async def list_masters_missing_embeddings(self) -> list[MasterDrawing]:
        rows = (
            await self.session.scalars(
                select(MasterDrawing).where(MasterDrawing.embedding.is_(None))
            )
        ).all()
        return list(rows)

    async def update_master_embedding(self, master_key: str, embedding: list[float]) -> None:
        await self.session.execute(
            update(MasterDrawing)
            .where(MasterDrawing.master_key == master_key)
            .values(embedding=embedding)
        )

    async def search_masters_by_embedding(
        self, query_vector: list[float], limit: int = 20
    ) -> list[tuple[str, float]]:
        distance = MasterDrawing.embedding.cosine_distance(query_vector)
        rows = (
            await self.session.execute(
                select(MasterDrawing.master_key, (1 - distance).label("similarity"))
                .where(MasterDrawing.embedding.isnot(None))
                .order_by(distance)
                .limit(limit)
            )
        ).all()
        return [(row.master_key, float(row.similarity)) for row in rows]
