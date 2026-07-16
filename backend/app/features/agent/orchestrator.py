import logging
from pathlib import Path

from app.config.settings import Settings
from app.core.models.schemas import AgentTraceStep, SketchAnalysis
from app.features.masters.loader import MasterRecord
from app.features.vision.sketch_analyzer import SketchAnalyzer, _filter_angle_lengths

logger = logging.getLogger(__name__)


class MatchOrchestrator:
    """
    The DL classifier alone picks the master (see MatchService.process_match).
    This orchestrator's only remaining job is reading handwritten dimension
    values off the sketch once the master is known — never choosing it.
    """

    def __init__(self, settings: Settings, analyzer: SketchAnalyzer):
        self.settings = settings
        self.analyzer = analyzer

    @staticmethod
    def _align_to_master(lengths: list[float], master_lengths: list[float]) -> list[float]:
        """Reorder extracted numbers so their size-rank matches the master's
        segment size-rank. Vision LLMs return numbers in arbitrary notice-order;
        a segment's dimension keeps its relative magnitude (feet small, top run
        large), so rank-matching restores the true left-to-right segment order."""
        if len(lengths) != len(master_lengths) or not lengths:
            return lengths
        master_rank = sorted(range(len(master_lengths)), key=lambda i: master_lengths[i])
        sorted_lengths = sorted(lengths)
        aligned = [0.0] * len(lengths)
        for rank, seg_idx in enumerate(master_rank):
            aligned[seg_idx] = sorted_lengths[rank]
        return aligned

    def extract_lengths(
        self, sketch_path: Path, master: MasterRecord, analysis: SketchAnalysis
    ) -> tuple[list[float], float, AgentTraceStep]:
        segment_count = master.segment_count
        filtered_lengths = _filter_angle_lengths(
            analysis.handwritten_lengths,
            analysis.angles_estimate,
        )
        angles_leaked = (
            analysis.handwritten_lengths
            and filtered_lengths != analysis.handwritten_lengths
        )

        # Use lengths from analyze when count matches and no angle values leaked in
        if (
            filtered_lengths
            and len(filtered_lengths) == segment_count
            and not angles_leaked
        ):
            aligned = self._align_to_master(filtered_lengths, master.drawing.lengths)
            logger.info(
                "orchestrator extract %s → using analyze lengths %s aligned to master → %s",
                sketch_path.name,
                filtered_lengths,
                aligned,
            )
            return aligned, analysis.confidence, AgentTraceStep(
                step="extract",
                status="completed",
                message=f"Extracted {len(aligned)} lengths, aligned to master segment order",
                data={
                    "extracted_lengths": aligned,
                    "confidence": analysis.confidence,
                    "source": "analyze",
                },
            )

        # Fallback: focused re-read when count is wrong or angle annotations were mixed in
        reason = "angle annotations filtered" if angles_leaked else "length count mismatch"
        logger.info(
            "orchestrator extract %s → fallback LLM (%s: have %d lengths, need %d segments)",
            sketch_path.name,
            reason,
            len(filtered_lengths),
            segment_count,
        )
        lengths, confidence = self.analyzer.extract_lengths(sketch_path, segment_count)

        if not lengths and filtered_lengths:
            lengths = filtered_lengths
            confidence = analysis.confidence
        elif not lengths and analysis.handwritten_lengths:
            lengths = analysis.handwritten_lengths
            confidence = analysis.confidence

        lengths = self._align_to_master(lengths, master.drawing.lengths)
        return lengths, confidence, AgentTraceStep(
            step="extract",
            status="completed",
            message=f"Extracted {len(lengths)} dimension values, aligned to master segment order",
            data={"extracted_lengths": lengths, "confidence": confidence, "source": "extract_fallback"},
        )
