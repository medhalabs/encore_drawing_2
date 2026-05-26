from pathlib import Path

from app.config.settings import Settings
from app.core.models.schemas import AgentTraceStep, CompareResult, ScoreBreakdown, SketchAnalysis
from app.features.agent.prompts import SELECT_MASTER_PROMPT
from app.features.db.database_service import db_service
from app.features.embeddings.service import EmbeddingService
from app.features.feedback.store import FeedbackStore
from app.features.masters.loader import MasterRecord
from app.features.ollama.client import OllamaService
from app.features.rag.retriever import MasterRetriever, RetrievalCandidate
from app.features.vision.profile_comparator import ProfileComparator
from app.features.vision.sketch_analyzer import SketchAnalyzer


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

        candidates = self.retriever.retrieve(analysis, top_k=5)
        return candidates, AgentTraceStep(
            step="retrieve",
            status="completed",
            message=f"Retrieved {len(candidates)} candidate masters (hybrid pgvector + rules)",
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

    def compare_candidates(
        self, sketch_path: Path, candidates: list[RetrievalCandidate]
    ) -> tuple[list[CompareResult], AgentTraceStep]:
        results: list[CompareResult] = []
        for candidate in candidates:
            vision = self.comparator.compare(sketch_path, candidate.master)
            combined = candidate.score / 100 * 0.35 + vision.score * 0.65
            results.append(
                CompareResult(
                    master_key=vision.master_key,
                    score=combined,
                    reasoning=f"retrieval={candidate.score:.0f}, vision={vision.score:.2f}: {vision.reasoning}",
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results, AgentTraceStep(
            step="compare",
            status="completed",
            message=f"Compared sketch against {len(results)} master drawings",
            data={
                "comparisons": [
                    {
                        "master_key": r.master_key,
                        "combined_score": round(r.score, 3),
                        "reasoning": r.reasoning,
                    }
                    for r in results
                ]
            },
        )

    def select_master(
        self,
        analysis: SketchAnalysis,
        candidates: list[RetrievalCandidate],
        comparisons: list[CompareResult],
    ) -> tuple[MasterRecord, float, ScoreBreakdown, list[str], AgentTraceStep]:
        warnings: list[str] = []
        if not comparisons:
            raise ValueError("No comparison results available")

        best_comparison = comparisons[0]
        best_key = best_comparison.master_key
        best = next(c.master for c in candidates if c.master.key == best_key)

        retrieval_score = next(c.score for c in candidates if c.master.key == best_key)
        vector_score = 0.0
        for reason in next(c.reasons for c in candidates if c.master.key == best_key):
            if reason.startswith("vector_sim="):
                try:
                    vector_score = float(reason.split("=", 1)[1])
                except ValueError:
                    vector_score = 0.0
                break

        vision_part = best_comparison.reasoning
        vision_score = 0.0
        if "vision=" in vision_part:
            try:
                vision_score = float(vision_part.split("vision=")[1].split(":")[0])
            except ValueError:
                vision_score = 0.0

        if vision_score < self.settings.min_vision_score:
            warnings.append(
                f"Low shape match (vision {vision_score:.0%}). Please verify or use Correct this match."
            )

        feedback_boost = 0.0
        for c in candidates:
            if c.master.key == best_key:
                for reason in c.reasons:
                    if reason.startswith("feedback"):
                        feedback_boost = max(feedback_boost, 50.0)

        breakdown = ScoreBreakdown(
            retrieval_score=round(retrieval_score, 1),
            vector_score=round(vector_score, 3),
            vision_score=round(vision_score, 3),
            feedback_boost=feedback_boost,
            combined_score=round(best_comparison.score, 3),
        )
        self._last_score_breakdown = breakdown

        confidence = vision_score if vision_score >= self.settings.min_vision_score else best_comparison.score * 0.7

        return best, confidence, breakdown, warnings, AgentTraceStep(
            step="match",
            status="warning" if warnings else "completed",
            message=f"Selected master {best.key} (vision {vision_score:.0%}, retrieval {retrieval_score:.0f})",
            data={
                "master_key": best.key,
                "confidence": confidence,
                "score_breakdown": breakdown.model_dump(),
            },
        )

    def extract_lengths(
        self, sketch_path: Path, master: MasterRecord, analysis: SketchAnalysis
    ) -> tuple[list[float], float, AgentTraceStep]:
        segment_count = master.segment_count
        lengths, confidence = self.analyzer.extract_lengths(sketch_path, segment_count)

        if not lengths and analysis.handwritten_lengths:
            lengths = analysis.handwritten_lengths
            confidence = analysis.confidence

        if len(lengths) != segment_count and analysis.handwritten_lengths:
            if len(analysis.handwritten_lengths) == segment_count:
                lengths = analysis.handwritten_lengths

        return lengths, confidence, AgentTraceStep(
            step="extract",
            status="completed",
            message=f"Extracted {len(lengths)} dimension values",
            data={"extracted_lengths": lengths, "confidence": confidence},
        )
