#!/usr/bin/env python3
"""Evaluate match pipeline against Client_handwritten_data screenshots."""

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings
from app.features.agent.orchestrator import MatchOrchestrator
from app.features.feedback.store import FeedbackStore
from app.features.masters.catalog import MasterCatalog
from app.features.ollama.client import OllamaService
from app.features.rag.retriever import MasterRetriever
from app.features.vision.profile_comparator import ProfileComparator
from app.features.vision.sketch_analyzer import SketchAnalyzer
from app.services.match_service import MatchService


def normalize_timestamp(name: str) -> str | None:
    match = re.search(r"(\d{1,2}\.\d{2}\.\d{2})\s*(AM|PM)", name, re.I)
    if match:
        return f"{match.group(1)} {match.group(2).upper()}"
    return None


def find_ground_truth(ts: str, sketches_dir: Path) -> Path | None:
    for p in sketches_dir.glob("*.json"):
        if ts.replace(" ", "") in p.name.replace("\u202f", "").replace(" ", ""):
            return p
        if ts in p.name:
            return p
    return None


async def main():
    settings = get_settings()
    root = settings.master_drawings_dir.parents[1]
    test_dir = root / "testing" / "Client_handwritten_data"
    gt_dir = root / "user_drawings" / "Client_sketches"

    catalog = MasterCatalog(settings)
    catalog.load()
    feedback_store = FeedbackStore(settings, catalog)
    feedback_store.load()
    ollama = OllamaService(settings)
    retriever = MasterRetriever(catalog)
    retriever.set_feedback_entries(feedback_store.entries)
    orchestrator = MatchOrchestrator(
        settings,
        SketchAnalyzer(ollama),
        retriever,
        ProfileComparator(ollama),
        ollama,
        feedback_store,
    )
    service = MatchService(settings, catalog, orchestrator)

    images = sorted(test_dir.glob("*.png"))
    correct = 0
    total = 0

    for img in images[:5]:
        ts = normalize_timestamp(img.name)
        gt_path = find_ground_truth(ts, gt_dir) if ts else None
        gt_id = None
        if gt_path:
            with gt_path.open() as f:
                gt_id = json.load(f).get("_id")

        result = await service.process_match(img, img.name)
        total += 1
        matched_id = result.matched_master.id
        ok = gt_id and matched_id == gt_id
        if ok:
            correct += 1
        vision = result.score_breakdown.vision_score if result.score_breakdown else 0
        print(f"{img.name}: matched={result.matched_master.key} gt_id_match={ok} vision={vision:.0%}")

    print(f"\nMaster ID accuracy: {correct}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
