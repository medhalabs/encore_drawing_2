import json
from dataclasses import dataclass, field
from pathlib import Path

from app.core.models.schemas import EncoreDrawing


@dataclass
class ProfileFingerprint:
    """Deterministic geometric signature computed from exact master JSON data."""
    segment_count: int
    angle_pattern: tuple[int, ...]      # angles rounded to nearest 15°
    angle_pattern_rev: tuple[int, ...]  # reversed — for flip-invariant matching
    length_ratios: tuple[float, ...]    # normalised segment proportions
    direction: str
    has_start_fold: bool
    has_end_fold: bool
    part_class: str

    def angle_distance(self, sketch_angles: list[float]) -> float:
        """Min of forward and reversed angle distance — handles mirrored profiles."""
        if not sketch_angles or len(sketch_angles) != len(self.angle_pattern):
            return 999.0
        forward = sum(abs(a - b) for a, b in zip(self.angle_pattern, sketch_angles)) / len(sketch_angles)
        backward = sum(abs(a - b) for a, b in zip(self.angle_pattern_rev, sketch_angles)) / len(sketch_angles)
        return min(forward, backward)

    def length_ratio_distance(self, sketch_lengths: list[float]) -> float:
        """Compare relative proportions of segment lengths — ignores absolute scale."""
        if not sketch_lengths or len(sketch_lengths) != len(self.length_ratios):
            return 999.0
        total = sum(sketch_lengths)
        if total == 0:
            return 999.0
        sketch_ratios = [l / total for l in sketch_lengths]
        return sum(abs(a - b) for a, b in zip(self.length_ratios, sketch_ratios)) / len(sketch_ratios)


def _build_fingerprint(drawing: EncoreDrawing) -> ProfileFingerprint:
    angles = drawing.angles
    lengths = drawing.lengths
    total = sum(lengths) or 1.0
    angle_pattern = tuple(round(a / 15) * 15 for a in angles)
    return ProfileFingerprint(
        segment_count=len(lengths),
        angle_pattern=angle_pattern,
        angle_pattern_rev=tuple(reversed(angle_pattern)),
        length_ratios=tuple(l / total for l in lengths),
        direction=drawing.direction,
        has_start_fold=drawing.start_fold_type is not None,
        has_end_fold=drawing.end_fold_type is not None,
        part_class=drawing.part_class,
    )


@dataclass
class MasterRecord:
    key: str
    category: str
    basename: str
    json_path: Path
    image_path: Path
    drawing: EncoreDrawing
    fingerprint: ProfileFingerprint = field(init=False)

    def __post_init__(self) -> None:
        self.fingerprint = _build_fingerprint(self.drawing)

    @property
    def segment_count(self) -> int:
        return len(self.drawing.lengths)

    @property
    def display_name(self) -> str:
        if self.drawing.name.strip():
            return self.drawing.name.strip()
        return self.basename


def load_master_record(category: str, json_path: Path) -> MasterRecord | None:
    basename = json_path.stem
    image_path = json_path.with_suffix(".png")
    if not image_path.exists():
        return None

    with json_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    drawing = EncoreDrawing.model_validate(raw)
    key = f"{category}/{basename}"

    return MasterRecord(
        key=key,
        category=category,
        basename=basename,
        json_path=json_path,
        image_path=image_path,
        drawing=drawing,
    )


def load_all_masters(root_dir: Path) -> list[MasterRecord]:
    masters: list[MasterRecord] = []
    if not root_dir.exists():
        return masters

    for category_dir in sorted(root_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for json_path in sorted(category_dir.glob("*.json")):
            record = load_master_record(category_dir.name, json_path)
            if record:
                masters.append(record)

    return masters
