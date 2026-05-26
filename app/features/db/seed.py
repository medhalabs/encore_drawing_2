from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.db.models import Correction, MasterDrawing
from app.features.embeddings.service import EmbeddingService
from app.features.masters.catalog import MasterCatalog


async def seed_masters_if_empty(session: AsyncSession, catalog: MasterCatalog) -> int:
    count = await session.scalar(select(func.count()).select_from(MasterDrawing))
    if count and count > 0:
        return 0

    catalog.load()
    added = 0
    for master in catalog.masters:
        d = master.drawing
        session.add(
            MasterDrawing(
                encore_id=d.id,
                master_key=master.key,
                category=master.category,
                name=master.display_name,
                json_template=d.to_encore_dict(),
                image_path=str(master.image_path),
                segment_count=master.segment_count,
                part_class=d.part_class,
                angles=d.angles,
                fingerprint=catalog.fingerprint(master),
            )
        )
        added += 1
    await session.commit()
    return added


async def import_corrections_from_manifest(session: AsyncSession, entries: list) -> int:
    from app.core.models.schemas import FeedbackEntry

    added = 0
    for raw in entries:
        entry = raw if isinstance(raw, FeedbackEntry) else FeedbackEntry.model_validate(raw)
        existing = await session.get(Correction, entry.feedback_id)
        if existing:
            continue
        label = {}
        label_path = entry.label_path
        try:
            from pathlib import Path
            from app.config.settings import get_settings
            p = get_settings().feedback_path / label_path
            if p.exists():
                import json
                label = json.loads(p.read_text())
        except Exception:
            label = {"lengths": entry.lengths}
        session.add(
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
                label_json=label,
                previous_master_key=entry.previous_master_key,
            )
        )
        added += 1
    if added:
        await session.commit()
    return added


async def backfill_master_embeddings(
    session: AsyncSession,
    catalog: MasterCatalog,
    embedding_service: EmbeddingService,
) -> int:
    catalog.load()
    repo_rows = (
        await session.scalars(
            select(MasterDrawing).where(MasterDrawing.embedding.is_(None))
        )
    ).all()
    if not repo_rows:
        return 0

    by_key = {m.key: m for m in catalog.masters}
    updated = 0
    for row in repo_rows:
        master = by_key.get(row.master_key)
        if not master:
            continue
        vector = embedding_service.embed_master(master)
        row.embedding = vector
        updated += 1

    if updated:
        await session.commit()
    return updated
