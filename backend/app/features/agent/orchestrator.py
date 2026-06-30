import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from app.config.settings import Settings
from app.core.models.schemas import AgentTraceStep, CompareResult, ScoreBreakdown, SketchAnalysis, TopCandidate
from app.features.agent.prompts import SELECT_MASTER_PROMPT
from app.features.db.database_service import db_service
from app.features.embeddings.service import EmbeddingService
from app.features.feedback.store import FeedbackStore
from app.features.masters.loader import MasterRecord
from app.features.ollama.client import OllamaService
from app.features.rag.retriever import MasterRetriever, RetrievalCandidate
from app.features.vision.profile_comparator import ProfileComparator
from app.features.vision.sketch_analyzer import SketchAnalyzer, _filter_angle_lengths

logger = logging.getLogger(__name__)


@dataclass
class _CompareDetail:
    """Internal: carries raw retrieval + vision scores without string encoding."""
    master: MasterRecord
    retrieval_score: float
    vision_score: float
    vector_score: float
    combined_score: float
    reasoning: str
    feedback_boost: float


class MatchOrchestrator:
    def __init__(
        self,
        settings: Settings,
        analyzer: SketchAnalyzer,
        retriever: MasterRetriever,
        comparator: ProfileComparator,
        ollama: OllamaService,
        feedback_store: FeedbackStore,
        embedding_service: EmbeddingService,
    ):
        self.settings = settings
        self.analyzer = analyzer
        self.retriever = retriever
        self.comparator = comparator
        self.ollama = ollama
        self.feedback_store = feedback_store
        self.embedding_service = embedding_service
        self._last_score_breakdown: ScoreBreakdown | None = None
        self._last_compare_details: list[_CompareDetail] = []

    @property
    def last_score_breakdown(self) -> ScoreBreakdown | None:
        return self._last_score_breakdown

    def _match_feedback_images(self, sketch_path: Path) -> dict[str, float]:
        boosts: dict[str, float] = {}
        for entry in self.feedback_store.entries:
            ref_path = self.settings.feedback_path / entry.image_path
            if not ref_path.exists():
                continue
            result = self.comparator.compare_to_reference_image(sketch_path, ref_path)
            if result.score >= self.settings.feedback_image_match_threshold:
                boosts[entry.master_key] = max(
                    boosts.get(entry.master_key, 0.0),
                    self.settings.feedback_image_boost * result.score,
                )
        return boosts

    def analyze_sketch(self, sketch_path: Path) -> tuple[SketchAnalysis, AgentTraceStep]:
        logger.info("orchestrator analyze_sketch %s", sketch_path.name)
        analysis = self.analyzer.analyze(sketch_path)
        return analysis, AgentTraceStep(
            step="analyze",
            status="completed",
            message=f"Detected {analysis.segment_count} segments, part hint: {analysis.part_class_hint or 'unknown'}",
            data={
                "segment_count": analysis.segment_count,
                "part_class_hint": analysis.part_class_hint,
                "handwritten_lengths": analysis.handwritten_lengths,
                "confidence": analysis.confidence,
            },
        )

    async def retrieve_candidates(
        self, sketch_path: Path, analysis: SketchAnalysis
    ) -> tuple[list[RetrievalCandidate], AgentTraceStep]:
        image_boosts = self._match_feedback_images(sketch_path)
        self.retriever.set_image_boosts(image_boosts)

        sketch_embed_text = self.embedding_service.build_sketch_embed_text(analysis)
        vector_scores: dict[str, float] = {}
        vector_top_candidates: list[dict] = []
        if db_service.enabled:
            try:
                sketch_vector = self.embedding_service.embed_text(sketch_embed_text)
                vector_scores = await db_service.search_masters_by_embedding(
                    sketch_vector, limit=self.settings.vector_search_top_k
                )
                vector_top_candidates = [
                    {"key": key, "similarity": round(sim, 3)}
                    for key, sim in sorted(
                        vector_scores.items(), key=lambda item: item[1], reverse=True
                    )[:10]
                ]
            except Exception as e:
                vector_top_candidates = [{"error": str(e)}]
        self.retriever.set_vector_scores(vector_scores)

        candidates = self.retriever.retrieve(analysis, top_k=3)
        logger.info(
            "orchestrator retrieve %s → %d candidates: %s",
            sketch_path.name,
            len(candidates),
            [c.master.key for c in candidates],
        )
        return candidates, AgentTraceStep(
            step="retrieve",
            status="completed",
            message=f"Retrieved {len(candidates)} candidate masters (hybrid pgvector + fingerprint + rules)",
            data={
                "candidates": [
                    {"key": c.master.key, "score": c.score, "reasons": c.reasons}
                    for c in candidates
                ],
                "feedback_image_boosts": image_boosts,
                "vector_top_candidates": vector_top_candidates,
                "sketch_embed_preview": sketch_embed_text[:240],
            },
        )

    def _compare_one(self, sketch_path: Path, candidate: RetrievalCandidate) -> _CompareDetail:
        vision = self.comparator.compare(sketch_path, candidate.master)
        vector_score = 0.0
        for reason in candidate.reasons:
            if reason.startswith("vector_sim="):
                try:
                    vector_score = float(reason.split("=", 1)[1])
                except ValueError:
                    pass
                break
        feedback_boost = sum(50.0 for r in candidate.reasons if r.startswith("feedback"))
        combined = candidate.score / 100 * 0.35 + vision.score * 0.65
        return _CompareDetail(
            master=candidate.master,
            retrieval_score=candidate.score,
            vision_score=vision.score,
            vector_score=vector_score,
            combined_score=combined,
            reasoning=vision.reasoning,
            feedback_boost=min(feedback_boost, 50.0),
        )

    async def compare_candidates(
        self, sketch_path: Path, candidates: list[RetrievalCandidate]
    ) -> tuple[list[_CompareDetail], AgentTraceStep]:
        # Run all vision comparisons in parallel — each LLM call runs in its own thread
        # so they overlap on the network I/O instead of waiting serially.
        # gemma4:31b-cloud handles concurrent requests; capped at 3 workers (top_k=3).
        loop = asyncio.get_event_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(self._compare_one, sketch_path, c) for c in candidates]
            details = await loop.run_in_executor(
                None, lambda: [f.result() for f in concurrent.futures.as_completed(futures)]
            )

        details.sort(key=lambda d: d.combined_score, reverse=True)
        self._last_compare_details = details
        logger.info(
            "orchestrator compare %s → %s",
            sketch_path.name,
            [(d.master.key, round(d.combined_score, 3), round(d.vision_score, 3)) for d in details],
        )

        return details, AgentTraceStep(
            step="compare",
            status="completed",
            message=f"Compared sketch against {len(details)} master drawings (parallel)",
            data={
                "comparisons": [
                    {
                        "master_key": d.master.key,
                        "combined_score": round(d.combined_score, 3),
                        "vision_score": round(d.vision_score, 3),
                        "retrieval_score": round(d.retrieval_score, 1),
                        "reasoning": d.reasoning,
                    }
                    for d in details
                ]
            },
        )

    def select_master(
        self,
        analysis: SketchAnalysis,
        candidates: list[RetrievalCandidate],
        details: list[_CompareDetail],
    ) -> tuple[MasterRecord | None, float, ScoreBreakdown, list[str], list[TopCandidate], AgentTraceStep]:
        warnings: list[str] = []
        if not details:
            raise ValueError("No comparison results available")

        best = details[0]
        vision_score = best.vision_score

        # Reject match entirely when no candidate scores high enough
        no_match_threshold = getattr(self.settings, "no_match_vision_threshold", 0.55)
        if vision_score < no_match_threshold:
            logger.info(
                "orchestrator select %s → no_match best=%s vision=%.2f threshold=%.2f",
                analysis.part_class_hint or "unknown",
                best.master.key,
                vision_score,
                no_match_threshold,
            )
            top_candidates = [
                TopCandidate(
                    key=d.master.key,
                    name=d.master.display_name,
                    category=d.master.category,
                    image_url=f"/api/v1/masters/{d.master.key}/image",
                    combined_score=round(d.combined_score, 3),
                    vision_score=round(d.vision_score, 3),
                    reasoning=d.reasoning,
                )
                for d in details[:3]
            ]
            breakdown = ScoreBreakdown(
                retrieval_score=round(best.retrieval_score, 1),
                vector_score=round(best.vector_score, 3),
                vision_score=round(vision_score, 3),
                feedback_boost=best.feedback_boost,
                combined_score=round(best.combined_score, 3),
            )
            self._last_score_breakdown = breakdown
            return None, 0.0, breakdown, [], top_candidates, AgentTraceStep(
                step="match",
                status="no_match",
                message=f"No matching master found — best vision score {vision_score:.0%} is below threshold {no_match_threshold:.0%}",
                data={
                    "master_key": None,
                    "best_candidate": best.master.key,
                    "vision_score": round(vision_score, 3),
                    "threshold": no_match_threshold,
                    "low_confidence": True,
                },
            )

        if vision_score < self.settings.min_vision_score:
            warnings.append(
                f"Low shape match (vision {vision_score:.0%}). Please verify or use Correct this match."
            )

        breakdown = ScoreBreakdown(
            retrieval_score=round(best.retrieval_score, 1),
            vector_score=round(best.vector_score, 3),
            vision_score=round(vision_score, 3),
            feedback_boost=best.feedback_boost,
            combined_score=round(best.combined_score, 3),
        )
        self._last_score_breakdown = breakdown

        confidence = vision_score if vision_score >= self.settings.min_vision_score else best.combined_score * 0.7

        # Top-3 candidates for low-confidence UI display
        top_candidates = [
            TopCandidate(
                key=d.master.key,
                name=d.master.display_name,
                category=d.master.category,
                image_url=f"/api/v1/masters/{d.master.key}/image",
                combined_score=round(d.combined_score, 3),
                vision_score=round(d.vision_score, 3),
                reasoning=d.reasoning,
            )
            for d in details[:3]
        ]

        logger.info(
            "orchestrator select → master=%s vision=%.2f combined=%.2f",
            best.master.key,
            vision_score,
            best.combined_score,
        )
        return best.master, confidence, breakdown, warnings, top_candidates, AgentTraceStep(
            step="match",
            status="warning" if warnings else "completed",
            message=f"Selected master {best.master.key} (vision {vision_score:.0%}, retrieval {best.retrieval_score:.0f})",
            data={
                "master_key": best.master.key,
                "confidence": confidence,
                "score_breakdown": breakdown.model_dump(),
                "low_confidence": vision_score < self.settings.min_vision_score,
            },
        )

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
            logger.info(
                "orchestrator extract %s → using analyze lengths %s (no fallback)",
                sketch_path.name,
                filtered_lengths,
            )
            return filtered_lengths, analysis.confidence, AgentTraceStep(
                step="extract",
                status="completed",
                message=f"Extracted {len(filtered_lengths)} lengths from analyze step (no extra LLM call)",
                data={
                    "extracted_lengths": filtered_lengths,
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

        return lengths, confidence, AgentTraceStep(
            step="extract",
            status="completed",
            message=f"Extracted {len(lengths)} dimension values (fallback LLM call)",
            data={"extracted_lengths": lengths, "confidence": confidence, "source": "extract_fallback"},
        )
