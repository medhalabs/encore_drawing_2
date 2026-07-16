import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from app.core.models.schemas import AgentTraceStep, MatchResult, MatchedMaster, SketchAnalysis
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

        logger.info("process_match start job=%s file=%s", job_id, original_filename)

        upload_step = AgentTraceStep(
            step="upload",
            status="completed",
            message=f"Saved upload as job {job_id}",
            data={"job_id": job_id, "filename": original_filename},
        )
        await self._emit(upload_step, on_step)

        # ── OpenCV preprocessing ────────────────────────────────────────
        from app.features.vision.sketch_preprocessor import preprocess_sketch
        preprocessed_pil = preprocess_sketch(sketch_path)

        # Save preprocessed image next to the upload so the frontend can display it
        preprocessed_dir = self.settings.upload_path / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_path = preprocessed_dir / f"{job_id}.png"
        preprocessed_pil.save(preprocessed_path)

        preprocess_step = AgentTraceStep(
            step="preprocess",
            status="completed",
            message="OpenCV: ruled lines removed, sketch isolated",
            data={
                "original_image_url": f"/api/v1/match/{job_id}/upload",
                "preprocessed_image_url": f"/api/v1/match/{job_id}/preprocessed",
            },
        )
        trace.append(preprocess_step)
        await self._emit(preprocess_step, on_step)

        # ── EfficientNet classifier — the DL alone picks the master ─────
        from app.core.models.schemas import ScoreBreakdown
        clf_result = self.classifier.predict_from_pil(preprocessed_pil) if self.classifier else None

        fast_master = self.catalog.get_by_key(clf_result.master_key) if clf_result else None
        fast_confidence = clf_result.confidence if clf_result else 0.0

        if fast_master is not None:
            if fast_confidence < self.settings.dl_review_threshold:
                warnings.append(
                    f"Low classifier confidence ({fast_confidence:.0%}) — please review this match."
                )
            logger.info(
                "process_match %s → %s (%.0f%%)",
                job_id,
                fast_master.key,
                fast_confidence * 100,
            )
            clf_step = AgentTraceStep(
                step="classify",
                status="completed",
                message=f"EfficientNet → {fast_master.key} ({fast_confidence:.0%})",
                data={"master_key": fast_master.key, "confidence": fast_confidence, "source": "efficientnet"},
            )
            trace.append(clf_step)
            await self._emit(clf_step, on_step)

            # The DL pick is authoritative — skip the generic analyze (which
            # re-guesses category/segment count and often contradicts the DL).
            # One focused call: read exactly the master's segment lengths.
            lengths, extract_conf = await asyncio.to_thread(
                self.orchestrator.analyzer.extract_lengths,
                sketch_path,
                fast_master.segment_count,
            )
            analysis = SketchAnalysis(
                segment_count=fast_master.segment_count,
                angles_estimate=[],
                handwritten_lengths=lengths,
                part_class_hint=fast_master.category,
                fold_hints="",
                confidence=extract_conf,
                description=f"{fast_master.key} (selected by classifier)",
            )
            step = AgentTraceStep(
                step="analyze",
                status="completed",
                message=(
                    f"Read {len(lengths)} segment lengths for {fast_master.key} "
                    f"({fast_master.segment_count} segments expected)"
                ),
                data={
                    "segment_count": fast_master.segment_count,
                    "part_class_hint": fast_master.category,
                    "handwritten_lengths": lengths,
                    "confidence": extract_conf,
                    "source": "master_constrained_read",
                },
            )
            trace.append(step)
            await self._emit(step, on_step)

            master = fast_master
            confidence = fast_confidence
            breakdown = ScoreBreakdown(
                retrieval_score=0.0,
                vector_score=0.0,
                vision_score=fast_confidence,
                feedback_boost=0.0,
                combined_score=fast_confidence,
            )
            top_candidates = []
        else:
            # Classifier returned nothing (model not loaded yet)
            warnings = ["Classifier not ready — no model loaded. Please retrain."]
            result = MatchResult(
                job_id=job_id,
                matched_master=None,
                no_match=True,
                confidence=0.0,
                extracted_lengths=[],
                filled_json={},
                agent_trace=trace,
                upload_image_url=f"/api/v1/match/{job_id}/upload",
                warnings=warnings,
                score_breakdown=ScoreBreakdown(0, 0, 0, 0, 0),
                top_candidates=[],
            )
            self._results[job_id] = result
            await db_service.save_match(result, str(sketch_path))
            return result

        lengths, extract_conf, step = await asyncio.to_thread(
            self.orchestrator.extract_lengths, sketch_path, master, analysis
        )
        trace.append(step)
        await self._emit(step, on_step)

        # Match confidence is the classifier's own — it made the pick, it owns the number.
        # extract_conf (length-reading confidence) is reported separately in the extract step.
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
        logger.info(
            "process_match %s done → master=%s confidence=%.3f lengths=%s",
            job_id,
            master.key,
            result.confidence,
            lengths,
        )
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
