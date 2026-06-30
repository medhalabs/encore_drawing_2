import logging
import statistics
from pathlib import Path

from app.core.models.schemas import SketchAnalysis
from app.features.ollama.client import OllamaService

logger = logging.getLogger(__name__)

LENGTH_VS_ANGLE_RULES = """
SEGMENT LENGTHS vs ANGLE ANNOTATIONS (critical):
- handwritten_lengths = ONLY straight-segment length dimensions (numbers written beside/along a line run).
- NEVER put angle measurements in handwritten_lengths.
- A number is an ANGLE annotation — skip it entirely — if ANY of these apply:
  • A degree symbol is present (°, a small circle/o above or beside the digits, or "deg")
  • A curved arc is drawn between two meeting lines at a corner/bend (standard angle notation)
  • The number sits at a corner junction rather than along a straight segment
- Put bend/angle values ONLY in angles_estimate.
- segment_count = count of straight segments that have a length dimension in handwritten_lengths (not angle annotations).
"""

ANALYZE_PROMPT = f"""Analyze this handwritten roofing/flashing profile sketch image.

Extract ALL of the following in ONE pass:

1. segment_count: number of straight segments in the cross-section profile. Count only segments whose LENGTH dimension you include in handwritten_lengths. Do NOT count curves, rounded corners, or junctions as separate segments. Do NOT count angle annotations.
2. angles_estimate: list of bend angles annotated on the sketch (numbers with ° or arc symbols at corners). Best effort.
3. handwritten_lengths: segment LENGTH numbers only, in drawing order (left-to-right or top-to-bottom). Read every length dimension — but exclude all angle annotations per rules below.
{LENGTH_VS_ANGLE_RULES}
4. part_class_hint: classify the profile shape into ONE of these categories:
   - Gutters: U-channel or trough shape, open at top, collects rainwater
   - Capping: hat/cap shape that sits over a ridge, usually symmetrical
   - Aprons: flat step or apron shape, typically an L or Z profile against a wall
   - RidgeValley: ridge or valley flashing with angled meeting faces
   - Soakers: small step flashings around individual tiles
   - FootMoulds: base/foot mould profiles
   - Misc: anything that doesn't fit above
5. fold_hints: any safety fold or hem annotations
6. confidence: 0.0 to 1.0 how confident you are in segment count and category
7. description: brief description of the profile shape including orientation (e.g. "Z-profile facing right, unequal legs")

Return ONLY valid JSON with keys: segment_count, angles_estimate, handwritten_lengths, part_class_hint, fold_hints, confidence, description
All lengths must be numeric (mm values). If a length is unreadable write 0. Never return an empty handwritten_lengths list if segment length numbers are visible."""


def _safe_float_list(values: list, label: str = "value") -> list[float]:
    """Parse numeric lists from LLM JSON, skipping garbage like category codes."""
    parsed: list[float] = []
    for x in values:
        if isinstance(x, (int, float)):
            parsed.append(float(x))
            continue
        if isinstance(x, str):
            cleaned = x.strip().replace("°", "").replace("deg", "").strip()
            try:
                parsed.append(float(cleaned))
            except ValueError:
                logger.warning("skipping non-numeric %s: %r", label, x)
        else:
            logger.warning("skipping non-numeric %s: %r", label, x)
    return parsed


def _filter_angle_lengths(
    lengths: list[float],
    angles: list[float],
    tolerance: float = 1.0,
) -> list[float]:
    """Drop values that match angle annotations the model also reported."""
    if not lengths or not angles:
        return lengths
    filtered: list[float] = []
    for length in lengths:
        if any(abs(length - angle) <= tolerance for angle in angles):
            logger.info("filtering length %.1f — matches angle annotation", length)
            continue
        filtered.append(length)
    return filtered


def _parse_analysis(data: dict) -> SketchAnalysis:
    angles = _safe_float_list(data.get("angles_estimate", []), label="angle")
    lengths = _safe_float_list(data.get("handwritten_lengths", []), label="length")
    lengths = _filter_angle_lengths(lengths, angles)
    try:
        segment_count = int(data.get("segment_count", 0))
    except (TypeError, ValueError):
        segment_count = 0
    if lengths and len(lengths) != segment_count:
        segment_count = len(lengths)
    return SketchAnalysis(
        segment_count=segment_count,
        angles_estimate=angles,
        handwritten_lengths=lengths,
        part_class_hint=str(data.get("part_class_hint", "")),
        fold_hints=str(data.get("fold_hints", "")),
        confidence=float(data.get("confidence", 0.0)),
        description=str(data.get("description", "")),
    )


def _consensus_angles(results: list[SketchAnalysis], target_count: int) -> list[float]:
    """Median of angle estimates across multiple runs — reduces LLM noise."""
    valid = [r.angles_estimate for r in results if len(r.angles_estimate) == target_count]
    if not valid:
        # Fall back to any result with angles
        for r in results:
            if r.angles_estimate:
                return r.angles_estimate
        return []
    return [statistics.median(run[i] for run in valid) for i in range(target_count)]


def _consensus_segment_count(results: list[SketchAnalysis]) -> int:
    counts = [r.segment_count for r in results if r.segment_count > 0]
    if not counts:
        return 0
    # Use mode; tie-break toward majority
    return max(set(counts), key=counts.count)


class SketchAnalyzer:
    def __init__(self, ollama: OllamaService, consensus_runs: int = 3):
        self.ollama = ollama
        self.consensus_runs = consensus_runs

    def analyze(self, sketch_path: Path) -> SketchAnalysis:
        # Use the fast analyze model (gemma3:4b) — speed matters here, accuracy comes from compare step
        analyze_model = getattr(self.ollama.settings, "ollama_analyze_model", "")
        logger.info(
            "analyze %s model=%s consensus_runs=%d",
            sketch_path.name,
            analyze_model or self.ollama.settings.ollama_vision_model,
            self.consensus_runs,
        )
        results: list[SketchAnalysis] = []
        for _ in range(self.consensus_runs):
            data = self.ollama.chat_vision_json(
                ANALYZE_PROMPT,
                [sketch_path],
                system=(
                    "You are an expert at reading handwritten engineering sketches for metal flashing profiles. "
                    "You never confuse angle annotations (numbers with ° or arc symbols at bends) "
                    "with segment length dimensions written along straight runs."
                ),
                model=analyze_model,
            )
            results.append(_parse_analysis(data))

        # Consensus segment count (mode)
        segment_count = _consensus_segment_count(results)

        # Best confidence run drives the rest of the metadata
        best = max(results, key=lambda r: r.confidence)

        # Ground truth: if handwritten lengths were found, trust their count over
        # the visual segment count (LLM often over-counts curves/corners as segments)
        if best.handwritten_lengths and len(best.handwritten_lengths) != segment_count:
            segment_count = len(best.handwritten_lengths)

        # Consensus angles (median per position, flip-invariant handled in retriever)
        angles = _consensus_angles(results, segment_count - 1 if segment_count > 1 else 0)

        analysis = SketchAnalysis(
            segment_count=segment_count,
            angles_estimate=angles,
            handwritten_lengths=best.handwritten_lengths,
            part_class_hint=best.part_class_hint,
            fold_hints=best.fold_hints,
            confidence=best.confidence,
            description=best.description,
        )
        logger.info(
            "analyze %s → segments=%d lengths=%s hint=%s confidence=%.2f",
            sketch_path.name,
            analysis.segment_count,
            analysis.handwritten_lengths,
            analysis.part_class_hint,
            analysis.confidence,
        )
        return analysis

    def extract_lengths(self, sketch_path: Path, segment_count: int) -> tuple[list[float], float]:
        """
        Re-reads lengths from sketch with the segment count as a hint.
        Only called as a fallback when analyze() didn't return enough lengths.
        Avoids a second LLM call if Redis cache already has the analyze result.
        """
        from app.features.cache import redis_cache
        prompt = f"""Read the handwritten segment LENGTH dimensions on this sketch image.
The profile has {segment_count} straight segments. Return ONLY JSON:
{{"handwritten_lengths": [number, ...], "angles_estimate": [number, ...], "confidence": 0.0-1.0}}
{LENGTH_VS_ANGLE_RULES}
Order handwritten_lengths in drawing order. Use integers or decimals as written.
Return exactly {segment_count} segment lengths. Ignore every angle annotation."""
        logger.info(
            "extract_lengths fallback for %s segment_count=%d model=%s",
            sketch_path.name,
            segment_count,
            self.ollama.settings.ollama_vision_model,
        )
        data = self.ollama.chat_vision_json(
            prompt,
            [sketch_path],
            system=(
                "You extract segment length dimensions from engineering sketches. "
                "Numbers with degree symbols or arc notation at bends are angles — never lengths."
            ),
            model=self.ollama.settings.ollama_vision_model,
        )
        angles = _safe_float_list(data.get("angles_estimate", []), label="angle")
        lengths = _filter_angle_lengths(
            _safe_float_list(data.get("handwritten_lengths", []), label="length"),
            angles,
        )
        confidence = float(data.get("confidence", 0.0))
        logger.info(
            "extract_lengths %s → lengths=%s confidence=%.2f",
            sketch_path.name,
            lengths,
            confidence,
        )
        return lengths, confidence
