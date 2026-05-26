from typing import Any

from pydantic import BaseModel, Field


class EncoreDrawing(BaseModel):
    id: str = Field(alias="_id")
    name: str = ""
    lengths: list[float]
    angles: list[float]
    direction: str
    reverse_color: bool = Field(alias="reverseColor")
    is_taper: bool = Field(alias="isTaper")
    far_lengths: list[float] = Field(default_factory=list, alias="farLengths")
    far_angles: list[float] = Field(default_factory=list, alias="farAngles")
    near_lengths: list[float] = Field(default_factory=list, alias="nearLengths")
    near_angles: list[float] = Field(default_factory=list, alias="nearAngles")
    part_group: str = Field(alias="partGroup")
    part_class: str = Field(alias="partClass")
    first_segment_angle: float = Field(alias="firstSegmentAngle")
    flip_h: bool = Field(alias="flipH")
    flip_v: bool = Field(alias="flipV")
    start_fold_type: str | None = Field(default=None, alias="startFoldType")
    start_fold_direction: str | None = Field(default=None, alias="startFoldDirection")
    start_fold_length: float | None = Field(default=None, alias="startFoldLength")
    start_fold_gap: float | None = Field(default=None, alias="startFoldGap")
    end_fold_type: str | None = Field(default=None, alias="endFoldType")
    end_fold_direction: str | None = Field(default=None, alias="endFoldDirection")
    end_fold_length: float | None = Field(default=None, alias="endFoldLength")
    end_fold_gap: float | None = Field(default=None, alias="endFoldGap")

    model_config = {"populate_by_name": True}

    def to_encore_dict(self) -> dict[str, Any]:
        return {
            "_id": self.id,
            "name": self.name,
            "lengths": self.lengths,
            "angles": self.angles,
            "direction": self.direction,
            "reverseColor": self.reverse_color,
            "isTaper": self.is_taper,
            "farLengths": self.far_lengths,
            "farAngles": self.far_angles,
            "nearLengths": self.near_lengths,
            "nearAngles": self.near_angles,
            "partGroup": self.part_group,
            "partClass": self.part_class,
            "firstSegmentAngle": self.first_segment_angle,
            "flipH": self.flip_h,
            "flipV": self.flip_v,
            "startFoldType": self.start_fold_type,
            "startFoldDirection": self.start_fold_direction,
            "startFoldLength": self.start_fold_length,
            "startFoldGap": self.start_fold_gap,
            "endFoldType": self.end_fold_type,
            "endFoldDirection": self.end_fold_direction,
            "endFoldLength": self.end_fold_length,
            "endFoldGap": self.end_fold_gap,
        }


class MasterSummary(BaseModel):
    key: str
    id: str
    name: str
    category: str
    segment_count: int
    part_class: str
    image_url: str


class SketchAnalysis(BaseModel):
    segment_count: int
    angles_estimate: list[float] = Field(default_factory=list)
    handwritten_lengths: list[float] = Field(default_factory=list)
    part_class_hint: str = ""
    fold_hints: str = ""
    confidence: float = 0.0
    description: str = ""


class CompareResult(BaseModel):
    master_key: str
    score: float
    reasoning: str = ""


class AgentTraceStep(BaseModel):
    step: str
    status: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class MatchedMaster(BaseModel):
    key: str
    id: str
    name: str
    category: str
    image_url: str
    master_lengths: list[float]


class ScoreBreakdown(BaseModel):
    retrieval_score: float = 0.0
    vector_score: float = 0.0
    vision_score: float = 0.0
    feedback_boost: float = 0.0
    combined_score: float = 0.0


class MatchResult(BaseModel):
    job_id: str
    matched_master: MatchedMaster
    confidence: float
    extracted_lengths: list[float]
    filled_json: dict[str, Any]
    agent_trace: list[AgentTraceStep]
    upload_image_url: str = ""
    warnings: list[str] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown | None = None


class FeedbackRequest(BaseModel):
    job_id: str
    master_key: str
    lengths: list[float]
    note: str = ""


class FeedbackEntry(BaseModel):
    feedback_id: str
    job_id: str
    master_key: str
    master_id: str
    segment_count: int
    angles: list[float]
    part_class: str
    lengths: list[float]
    note: str = ""
    image_path: str
    label_path: str
    created_at: str
    previous_master_key: str = ""


class FeedbackResponse(BaseModel):
    entry: FeedbackEntry
    filled_json: dict[str, Any]
    message: str = "Correction saved. Future similar sketches will prefer this master."
