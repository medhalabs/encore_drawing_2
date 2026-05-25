import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.features.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MasterDrawing(Base):
    __tablename__ = "master_drawings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    encore_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    master_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    category: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(256), default="")
    json_template: Mapped[dict] = mapped_column(JSONB)
    image_path: Mapped[str] = mapped_column(Text)
    segment_count: Mapped[int] = mapped_column()
    part_class: Mapped[str] = mapped_column(String(64))
    angles: Mapped[list] = mapped_column(JSONB, default=list)
    fingerprint: Mapped[str] = mapped_column(Text, default="")
    embedding = mapped_column(Vector(768), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MatchJob(Base):
    __tablename__ = "match_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    upload_path: Mapped[str] = mapped_column(Text, default="")
    matched_master_key: Mapped[str] = mapped_column(String(128))
    matched_encore_id: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    extracted_lengths: Mapped[list] = mapped_column(JSONB)
    filled_json: Mapped[dict] = mapped_column(JSONB)
    score_breakdown: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    agent_trace: Mapped[list] = mapped_column(JSONB, default=list)
    warnings: Mapped[list] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Correction(Base):
    __tablename__ = "corrections"

    feedback_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), index=True)
    master_key: Mapped[str] = mapped_column(String(128), index=True)
    master_id: Mapped[str] = mapped_column(String(64))
    segment_count: Mapped[int] = mapped_column()
    angles: Mapped[list] = mapped_column(JSONB, default=list)
    part_class: Mapped[str] = mapped_column(String(64))
    lengths: Mapped[list] = mapped_column(JSONB)
    note: Mapped[str] = mapped_column(Text, default="")
    image_path: Mapped[str] = mapped_column(Text)
    label_json: Mapped[dict] = mapped_column(JSONB)
    previous_master_key: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
