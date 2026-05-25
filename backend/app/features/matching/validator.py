from app.core.models.schemas import EncoreDrawing


def validate_drawing(drawing: EncoreDrawing) -> list[str]:
    warnings: list[str] = []
    expected = len(drawing.angles) + 1
    if len(drawing.lengths) != expected:
        warnings.append(
            f"Length count ({len(drawing.lengths)}) does not match angles+1 ({expected})"
        )
    if any(l <= 0 for l in drawing.lengths):
        warnings.append("One or more lengths are zero or negative")
    return warnings
