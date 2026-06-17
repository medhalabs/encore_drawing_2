import statistics
from pathlib import Path

from app.core.models.schemas import SketchAnalysis
from app.features.ollama.client import OllamaService

ANALYZE_PROMPT = """Analyze this handwritten roofing/flashing profile sketch image.

Extract:
1. segment_count: number of straight segments in the cross-section profile
2. angles_estimate: list of bend angles between consecutive segments (best effort)
3. handwritten_lengths: list of dimension numbers written on the sketch, in segment order left-to-right or top-to-bottom
4. part_class_hint: e.g. Aprons, Gutters, Capping, RidgeValley, Soakers, FootMoulds, Misc
5. fold_hints: any safety fold or hem annotations
6. confidence: 0.0 to 1.0 how confident you are
7. description: brief description of the profile shape

Return ONLY valid JSON with keys: segment_count, angles_estimate, handwritten_lengths, part_class_hint, fold_hints, confidence, description
All lengths should be numeric (mm). If unreadable, use empty list for handwritten_lengths."""

EXTRACT_LENGTHS_PROMPT = """Read the handwritten dimension numbers on this sketch image.
The profile has {segment_count} segments. Return ONLY JSON:
{{"handwritten_lengths": [number, ...], "confidence": 0.0-1.0}}
Order lengths in drawing order. Use integers or decimals as written."""


def _parse_analysis(data: dict) -> SketchAnalysis:
    return SketchAnalysis(
        segment_count=int(data.get("segment_count", 0)),
        angles_estimate=[float(x) for x in data.get("angles_estimate", [])],
        handwritten_lengths=[float(x) for x in data.get("handwritten_lengths", [])],
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
        # Run analysis multiple times and merge — reduces LLM noise on angles + segment count
        results: list[SketchAnalysis] = []
        for _ in range(self.consensus_runs):
            data = self.ollama.chat_vision_json(
                ANALYZE_PROMPT,
                [sketch_path],
                system="You are an expert at reading handwritten engineering sketches for metal flashing profiles.",
            )
            results.append(_parse_analysis(data))

        # Consensus segment count (mode)
        segment_count = _consensus_segment_count(results)

        # Consensus angles (median per position, flip-invariant handled in retriever)
        angles = _consensus_angles(results, segment_count - 1 if segment_count > 1 else 0)

        # Best confidence run drives the rest of the metadata
        best = max(results, key=lambda r: r.confidence)

        return SketchAnalysis(
            segment_count=segment_count,
            angles_estimate=angles,
            handwritten_lengths=best.handwritten_lengths,
            part_class_hint=best.part_class_hint,
            fold_hints=best.fold_hints,
            confidence=best.confidence,
            description=best.description,
        )

    def extract_lengths(self, sketch_path: Path, segment_count: int) -> tuple[list[float], float]:
        prompt = EXTRACT_LENGTHS_PROMPT.format(segment_count=segment_count)
        data = self.ollama.chat_vision_json(prompt, [sketch_path])
        lengths = [float(x) for x in data.get("handwritten_lengths", [])]
        confidence = float(data.get("confidence", 0.0))
        return lengths, confidence
