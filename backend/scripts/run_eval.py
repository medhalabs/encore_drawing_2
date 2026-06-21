#!/usr/bin/env python3
"""
Standalone eval: runs every image in testing/Client_handwritten_data through the
match pipeline (no Postgres / Redis needed) and prints a full accuracy report.

Usage (from backend/):
    uv run python scripts/run_eval.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings
from app.features.agent.orchestrator import MatchOrchestrator
from app.features.embeddings.service import EmbeddingService
from app.features.feedback.store import FeedbackStore
from app.features.masters.catalog import MasterCatalog
from app.features.ollama.client import OllamaService
from app.features.rag.retriever import MasterRetriever
from app.features.vision.profile_comparator import ProfileComparator
from app.features.vision.sketch_analyzer import SketchAnalyzer
from app.services.match_service import MatchService

# ── monkey-patch db_service so it does nothing without Postgres ───────────────
from app.features.db import database_service as _db_mod

class _NoopDB:
    enabled = False
    async def save_match(self, *a, **kw): pass
    async def search_masters_by_embedding(self, *a, **kw): return {}
    async def load_corrections(self): return []
    async def startup(self, *a, **kw): pass
    async def shutdown(self): pass

_db_mod.db_service = _NoopDB()

# patch the reference used inside match_service too
import app.services.match_service as _ms_mod
_ms_mod.db_service = _NoopDB()


async def main():
    settings = get_settings()
    root = settings.master_drawings_dir.parents[1]
    test_dir = root / "testing" / "Client_handwritten_data"

    if not test_dir.exists():
        print(f"ERROR: test directory not found: {test_dir}")
        sys.exit(1)

    images = sorted(p for p in test_dir.glob("*.png"))
    print(f"Found {len(images)} test images in {test_dir}\n")

    # ── build pipeline ────────────────────────────────────────────────────────
    catalog = MasterCatalog(settings)
    catalog.load()

    feedback_store = FeedbackStore(settings, catalog)
    feedback_store.load()

    ollama = OllamaService(settings)
    embedding_service = EmbeddingService(settings, ollama, catalog)
    retriever = MasterRetriever(catalog)
    retriever.set_feedback_entries(feedback_store.entries)

    orchestrator = MatchOrchestrator(
        settings,
        SketchAnalyzer(ollama, consensus_runs=settings.analyzer_consensus_runs),
        retriever,
        ProfileComparator(ollama),
        ollama,
        feedback_store,
        embedding_service,
    )
    service = MatchService(settings, catalog, orchestrator)

    # ── run each image ────────────────────────────────────────────────────────
    rows = []
    errors = []

    for i, img in enumerate(images, 1):
        print(f"[{i:02d}/{len(images)}] {img.name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = await service.process_match(img, img.name)
            elapsed = time.time() - t0
            vision = result.score_breakdown.vision_score if result.score_breakdown else 0.0

            if result.no_match or result.matched_master is None:
                rows.append({
                    "img": img.name,
                    "matched_key": "NO_MATCH",
                    "matched_name": "No match",
                    "category": "—",
                    "confidence": 0.0,
                    "vision_score": vision,
                    "combined_score": 0.0,
                    "lengths": [],
                    "warnings": result.warnings,
                    "elapsed": elapsed,
                    "no_match": True,
                })
                print(f"NO_MATCH  (best vision={vision:.0%})  ({elapsed:.1f}s)")
            else:
                rows.append({
                    "img": img.name,
                    "matched_key": result.matched_master.key,
                    "matched_name": result.matched_master.name,
                    "category": result.matched_master.category,
                    "confidence": result.confidence,
                    "vision_score": vision,
                    "combined_score": result.score_breakdown.combined_score if result.score_breakdown else 0.0,
                    "lengths": result.extracted_lengths,
                    "warnings": result.warnings,
                    "elapsed": elapsed,
                    "no_match": False,
                })
                flag = "⚠️ LOW" if vision < 0.72 else "✓"
                print(f"{flag}  →  {result.matched_master.key}  vision={vision:.0%}  conf={result.confidence:.0%}  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            errors.append({"img": img.name, "error": str(e)})
            print(f"ERROR: {e}  ({elapsed:.1f}s)")

    # ── summary report ────────────────────────────────────────────────────────
    total = len(images)
    processed = len(rows)
    error_count = len(errors)

    matched_rows = [r for r in rows if not r.get("no_match")]
    no_match_rows = [r for r in rows if r.get("no_match")]
    high_conf = [r for r in matched_rows if r["vision_score"] >= 0.72]
    low_conf  = [r for r in matched_rows if r["vision_score"] < 0.72]

    print("\n" + "═" * 70)
    print("  EVAL REPORT — Encore Drawing Matcher")
    print("═" * 70)
    print(f"  Total images tested : {total}")
    print(f"  Matched (returned)  : {len(matched_rows)}")
    print(f"  No match returned   : {len(no_match_rows)}  (correctly rejected)")
    print(f"  Errors (crashed)    : {error_count}")
    print(f"  High confidence ≥72%: {len(high_conf)}  ({len(high_conf)/max(processed,1)*100:.1f}%)")
    print(f"  Low confidence <72% : {len(low_conf)}  ({len(low_conf)/max(processed,1)*100:.1f}%)")
    if rows:
        avg_t = sum(r["elapsed"] for r in rows) / len(rows)
        print(f"  Avg time / image    : {avg_t:.1f}s")
    if matched_rows:
        avg_v = sum(r["vision_score"] for r in matched_rows) / len(matched_rows)
        avg_c = sum(r["confidence"] for r in matched_rows) / len(matched_rows)
        print(f"  Avg vision score    : {avg_v:.1%}  (matched only)")
        print(f"  Avg confidence      : {avg_c:.1%}  (matched only)")

    # category breakdown
    from collections import Counter
    cat_counts = Counter(r["category"] for r in rows)
    print("\n  Matches by category:")
    for cat, cnt in cat_counts.most_common():
        print(f"    {cat:<25} {cnt}")

    # individual results table
    print("\n" + "─" * 70)
    print(f"  {'#':<3} {'Image':<52} {'Matched Key':<30} {'Vision':>7} {'OK?'}")
    print("─" * 70)
    for i, r in enumerate(rows, 1):
        ok = "✓" if r["vision_score"] >= 0.65 else "✗ LOW"
        name_short = r["img"][:50]
        print(f"  {i:<3} {name_short:<52} {r['matched_key']:<30} {r['vision_score']:>6.0%}  {ok}")

    # errors
    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    ✗ {e['img']}: {e['error']}")

    # low confidence details
    if low_conf:
        print("\n  LOW CONFIDENCE RESULTS (vision < 65%) — likely wrong template:")
        for r in low_conf:
            warn_str = "; ".join(r["warnings"]) if r["warnings"] else "—"
            print(f"    • {r['img'][:55]}")
            print(f"      → {r['matched_key']}  vision={r['vision_score']:.0%}  combined={r['combined_score']:.0%}")
            print(f"        warning: {warn_str}")

    print("\n" + "═" * 70)
    accuracy = len(high_conf) / max(processed, 1) * 100
    print(f"  ACCURACY (vision ≥ 65%): {len(high_conf)}/{processed} = {accuracy:.1f}%")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
