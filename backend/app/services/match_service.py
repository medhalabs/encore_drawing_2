import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from app.core.models.schemas import AgentTraceStep, MatchResult, MatchedMaster
from app.features.agent.orchestrator import MatchOrchestrator
from app.features.classifier.efficientnet import EfficientNetClassifier
from app.features.db.database_service import db_service
from app.features.masters.catalog import MasterCatalog
from app.features.matching.json_filler import fill_master_json
from app.features.matching.validator import validate_drawing
from app.config.settings import Settings

OnStepCallback = Callable[[AgentTraceStep], Awaitable[None]]

logger = logging.getLogger(__name__)


class MatchService:
    def __init__(
        self,
        settings: Settings,
        catalog: MasterCatalog,
        orchestrator: MatchOrchestrator,
        classifier: EfficientNetClassifier | None = None,
    ):
        self.settings = settings
        self.catalog = catalog
        self.orchestrator = orchestrator
        self.classifier = classifier
        self._results: dict[str, MatchResult] = {}

    def save_upload(self, filename: str, content: bytes) -> Path:
        upload_dir = self.settings.upload_path
        upload_dir.mkdir(parents=True, exist_ok=True)
        job_id = str(uuid.uuid4())
        ext = Path(filename).suffix or ".png"
        dest = upload_dir / f"{job_id}{ext}"
        dest.write_bytes(content)
        return dest

    def get_result(self, job_id: str) -> MatchResult | None:
        return self._results.get(job_id)

    async def _emit(self, step: AgentTraceStep, on_step: OnStepCallback | None) -> None:
        if on_step:
            await on_step(step)

    async def process_match(
        self,
        sketch_path: Path,
        original_filename: str,
        on_step: OnStepCallback | None = None,
    ) -> MatchResult:
        job_id = sketch_path.stem
        trace: list[AgentTraceStep] = []
        warnings: list[str] = []

        upload_step = AgentTraceStep(
            step="upload",
            status="completed",
            message=f"Saved upload as job {job_id}",
            data={"job_id": job_id, "filename": original_filename},
        )
        await self._emit(upload_step, on_step)

        # ── Fast path: EfficientNet classifier ─────────────────────────
        fast_master = None
        fast_confidence = 0.0
        if self.classifier is not None:
            clf_result = self.classifier.predict(sketch_path)
            if self.classifier.is_confident(clf_result):
                fast_master = self.catalog.get_by_key(clf_result.master_key)
                fast_confidence = clf_result.confidence

        if fast_master is not None:
            clf_step = AgentTraceStep(
                step="classify",
                status="completed",
                message=f"EfficientNet matched {fast_master.key} (confidence {fast_confidence:.0%}) — skipping LLM compare",
                data={"master_key": fast_master.key, "confidence": fast_confidence, "source": "efficientnet"},
            )
            trace.append(clf_step)
            await self._emit(clf_step, on_step)

            # Still run analyze so we get handwritten_lengths for extraction
            analysis, step = self.orchestrator.analyze_sketch(sketch_path)
            trace.append(step)
            await self._emit(step, on_step)

            master = fast_master
            confidence = fast_confidence
            from app.core.models.schemas import ScoreBreakdown
            breakdown = ScoreBreakdown(
                retrieval_score=0.0,
                vector_score=0.0,
                vision_score=fast_confidence,
                feedback_boost=0.0,
                combined_score=fast_confidence,
            )
            top_candidates = []
            warnings = []
        else:
            # ── Full LLM path ───────────────────────────────────────────
            analysis, step = self.orchestrator.analyze_sketch(sketch_path)
            trace.append(step)
            await self._emit(step, on_step)

            candidates, step = await self.orchestrator.retrieve_candidates(sketch_path, analysis)
            trace.append(step)
            await self._emit(step, on_step)

            if not candidates:
                raise ValueError("No master drawings found in catalog")

            comparisons, step = await self.orchestrator.compare_candidates(sketch_path, candidates)
            trace.append(step)
            await self._emit(step, on_step)

            master, confidence, breakdown, match_warnings, top_candidates, step = self.orchestrator.select_master(
                analysis, candidates, comparisons
            )
            trace.append(step)
            await self._emit(step, on_step)
            warnings.extend(match_warnings)

        # No match — return early with no_match=True
        if master is None:
            result = MatchResult(
                job_id=job_id,
                matched_master=None,
                no_match=True,
                confidence=0.0,
                extracted_lengths=[],
                filled_json={},
                agent_trace=trace,
                upload_image_url=f"/api/v1/match/{job_id}/upload",
                warnings=["No matching master drawing found for this sketch."],
                score_breakdown=breakdown,
                top_candidates=top_candidates,
            )
            self._results[job_id] = result
            await db_service.save_match(result, str(sketch_path))
            return result

        lengths, extract_conf, step = self.orchestrator.extract_lengths(
            sketch_path, master, analysis
        )
        trace.append(step)
        await self._emit(step, on_step)

        if breakdown.vision_score >= self.settings.min_vision_score:
            confidence = (breakdown.vision_score + extract_conf) / 2
        else:
            confidence = min(confidence, breakdown.combined_score)

        filled = fill_master_json(master, lengths)
        val_warnings = validate_drawing(filled)
        warnings.extend(val_warnings)

        validate_step = AgentTraceStep(
            step="validate",
            status="completed" if not val_warnings else "warning",
            message="Validated filled JSON output",
            data={"warnings": val_warnings},
        )
        trace.append(validate_step)
        await self._emit(validate_step, on_step)

        result = MatchResult(
            job_id=job_id,
            matched_master=MatchedMaster(
                key=master.key,
                id=master.drawing.id,
                name=master.display_name,
                category=master.category,
                image_url=f"/api/v1/masters/{master.key}/image",
                master_lengths=master.drawing.lengths,
            ),
            no_match=False,
            confidence=round(min(confidence, 1.0), 3),
            extracted_lengths=lengths,
            filled_json=filled.to_encore_dict(),
            agent_trace=trace,
            upload_image_url=f"/api/v1/match/{job_id}/upload",
            warnings=warnings,
            score_breakdown=breakdown,
            top_candidates=top_candidates,
        )
        self._results[job_id] = result
        await db_service.save_match(result, str(sketch_path))
        return result

    async def process_match_stream(
        self,
        sketch_path: Path,
        original_filename: str,
    ) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue = asyncio.Queue()

        async def on_step(step: AgentTraceStep) -> None:
            await queue.put({"type": "step", "payload": step.model_dump()})

        async def run() -> None:
            try:
                result = await self.process_match(sketch_path, original_filename, on_step=on_step)
                await queue.put({"type": "result", "payload": result.model_dump()})
            except Exception as e:
                logger.exception("Match pipeline failed for %s", original_filename)
                await queue.put({"type": "error", "payload": {"detail": str(e)}})
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        await task
