#!/usr/bin/env python3
"""
Standalone eval: runs every image in testing/Client_handwritten_data through the
match pipeline and compares against ground truth from eval_ground_truth.xlsx.

Usage (from backend/):
    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --gt /path/to/eval_ground_truth.xlsx
    uv run python scripts/run_eval.py --limit 10   # first 10 images only
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import get_settings
from app.features.agent.orchestrator import MatchOrchestrator
from app.features.classifier.efficientnet import EfficientNetClassifier
from app.features.classifier.retrain_service import RetrainService
from app.features.masters.catalog import MasterCatalog
from app.features.ollama.client import OllamaService
from app.features.vision.sketch_analyzer import SketchAnalyzer
from app.services.match_service import MatchService

# ── monkey-patch db_service so eval runs without Postgres ────────────────────
from app.features.db import database_service as _db_mod

class _NoopDB:
    enabled = False
    async def save_match(self, *a, **kw): pass
    async def search_masters_by_embedding(self, *a, **kw): return {}
    async def load_corrections(self): return []
    async def startup(self, *a, **kw): pass
    async def shutdown(self): pass

_db_mod.db_service = _NoopDB()
import app.services.match_service as _ms_mod
_ms_mod.db_service = _NoopDB()

NO_MATCH_SENTINEL = "no_match"


def load_ground_truth(xlsx_path: Path) -> dict[str, str]:
    """Returns {image_filename: correct_master_key_or_no_match}"""
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    gt: dict[str, str] = {}
    correct_col = [c for c in df.columns if "correct" in c.lower()][0]
    filename_col = [c for c in df.columns if "filename" in c.lower() or "image" in c.lower()][0]
    for _, row in df.iterrows():
        filename = str(row[filename_col]).strip()
        correct = str(row[correct_col]).strip()
        if not filename or filename == "nan":
            continue
        if "no match" in correct.lower() or correct == "nan":
            gt[filename] = NO_MATCH_SENTINEL
        else:
            gt[filename] = correct
    return gt


def is_correct(predicted_key: str, ground_truth_key: str) -> bool:
    """
    A match is correct if:
    - Both are no_match
    - Predicted key matches ground truth exactly
    - Predicted key matches ground truth ignoring -mirror suffix
      (e.g. Gutters/gutter-3-mirror matches Gutters/gutter-3)
    """
    if ground_truth_key == NO_MATCH_SENTINEL:
        return predicted_key == NO_MATCH_SENTINEL
    if predicted_key == NO_MATCH_SENTINEL:
        return False
    if predicted_key == ground_truth_key:
        return True
    # Accept mirror variant as correct if base key matches
    base_predicted = predicted_key.replace("-mirror", "")
    base_gt = ground_truth_key.replace("-mirror", "")
    return base_predicted == base_gt


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="/Users/pavanrajkg/Downloads/eval_ground_truth.xlsx",
                        help="Path to ground truth Excel file")
    parser.add_argument("--limit", type=int, default=0, help="Only run first N images (0 = all)")
    parser.add_argument("--out", default="", help="Write per-row CSV results to this file")
    args = parser.parse_args()

    settings = get_settings()
    root = settings.master_drawings_dir.parents[1]
    test_dir = root / "testing" / "Client_handwritten_data"

    if not test_dir.exists():
        print(f"ERROR: test directory not found: {test_dir}")
        sys.exit(1)

    images = sorted(p for p in test_dir.glob("*.png"))
    if args.limit:
        images = images[: args.limit]
    print(f"Found {len(images)} test images\n")

    # ── load ground truth ─────────────────────────────────────────────────────
    gt: dict[str, str] = {}
    gt_path = Path(args.gt)
    if gt_path.exists():
        gt = load_ground_truth(gt_path)
        print(f"Ground truth loaded: {len(gt)} entries from {gt_path.name}\n")
    else:
        print(f"WARNING: ground truth file not found at {gt_path} — accuracy vs GT will not be shown\n")

    # ── build pipeline ────────────────────────────────────────────────────────
    catalog = MasterCatalog(settings)
    catalog.load()
    print(f"Catalog: {len(catalog.masters)} masters loaded\n")

    ollama = OllamaService(settings)
    orchestrator = MatchOrchestrator(
        settings,
        SketchAnalyzer(ollama, consensus_runs=settings.analyzer_consensus_runs),
    )

    # DL classifier picks the master — mirrors app/main.py's startup wiring
    model_dir = Path(__file__).resolve().parents[1] / "data" / "models"
    label_index = sorted(m.key for m in catalog.masters)
    classifier = EfficientNetClassifier(model_dir, label_index)
    classifier._key_to_idx = {k: i for i, k in enumerate(label_index)}
    RetrainService(classifier, settings.master_drawings_dir, settings.feedback_path).update_class_counts()
    classifier.load_if_ready()

    service = MatchService(settings, catalog, orchestrator, classifier=classifier)

    # ── run each image ────────────────────────────────────────────────────────
    rows = []
    errors = []

    for i, img in enumerate(images, 1):
        gt_key = gt.get(img.name, "unknown")
        gt_label = gt_key if gt_key != "unknown" else "?"
        print(f"[{i:02d}/{len(images)}] {img.name[:55]}", end=" ", flush=True)
        t0 = time.time()
        try:
            result = await service.process_match(img, img.name)
            elapsed = time.time() - t0
            vision = result.score_breakdown.vision_score if result.score_breakdown else 0.0

            if result.no_match or result.matched_master is None:
                predicted = NO_MATCH_SENTINEL
                matched_key = "NO_MATCH"
            else:
                predicted = result.matched_master.key
                matched_key = predicted

            correct = is_correct(predicted, gt_key) if gt_key != "unknown" else None

            rows.append({
                "img": img.name,
                "predicted": matched_key,
                "ground_truth": gt_key,
                "correct": correct,
                "vision_score": vision,
                "confidence": result.confidence,
                "elapsed": elapsed,
                "no_match": result.no_match,
            })

            tick = "✓" if correct is True else ("✗" if correct is False else "?")
            no_match_str = " [NO_MATCH]" if predicted == NO_MATCH_SENTINEL else ""
            print(f"{tick}  {matched_key}{no_match_str}  gt={gt_label}  vision={vision:.0%}  ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            errors.append({"img": img.name, "error": str(e), "gt": gt_key})
            print(f"ERROR: {e}  ({elapsed:.1f}s)")

    # ── summary report ────────────────────────────────────────────────────────
    total = len(images)
    processed = len(rows)
    error_count = len(errors)
    rows_with_gt = [r for r in rows if r["ground_truth"] != "unknown"]
    correct_rows = [r for r in rows_with_gt if r["correct"] is True]
    wrong_rows   = [r for r in rows_with_gt if r["correct"] is False]

    matched_rows  = [r for r in rows if not r["no_match"]]
    no_match_rows = [r for r in rows if r["no_match"]]

    # Ground truth expected no-match
    gt_no_match = [r for r in rows_with_gt if r["ground_truth"] == NO_MATCH_SENTINEL]
    gt_has_match = [r for r in rows_with_gt if r["ground_truth"] != NO_MATCH_SENTINEL]
    correct_no_match = [r for r in gt_no_match if r["correct"]]
    correct_has_match = [r for r in gt_has_match if r["correct"]]

    print("\n" + "═" * 72)
    print("  EVAL REPORT — Encore Drawing Matcher")
    print("═" * 72)
    print(f"  Total images          : {total}")
    print(f"  Processed             : {processed}")
    print(f"  Errors (crashed)      : {error_count}")
    print(f"  With ground truth     : {len(rows_with_gt)}")
    print()
    if rows_with_gt:
        acc = len(correct_rows) / len(rows_with_gt) * 100
        print(f"  ✓ CORRECT             : {len(correct_rows)}/{len(rows_with_gt)} = {acc:.1f}%")
        print(f"  ✗ WRONG               : {len(wrong_rows)}/{len(rows_with_gt)}")
        print()
        if gt_has_match:
            shape_acc = len(correct_has_match) / len(gt_has_match) * 100
            print(f"  Shape accuracy        : {len(correct_has_match)}/{len(gt_has_match)} = {shape_acc:.1f}%  (images that HAVE a match)")
        if gt_no_match:
            nm_acc = len(correct_no_match) / len(gt_no_match) * 100
            print(f"  No-match accuracy     : {len(correct_no_match)}/{len(gt_no_match)} = {nm_acc:.1f}%  (images with no master)")
    print()
    print(f"  Returned a match      : {len(matched_rows)}")
    print(f"  Returned NO_MATCH     : {len(no_match_rows)}")
    if rows:
        avg_t = sum(r["elapsed"] for r in rows) / len(rows)
        print(f"  Avg time / image      : {avg_t:.1f}s")
    if matched_rows:
        avg_v = sum(r["vision_score"] for r in matched_rows) / len(matched_rows)
        print(f"  Avg vision score      : {avg_v:.1%}  (matched only)")

    # ── wrong results detail ──────────────────────────────────────────────────
    if wrong_rows:
        print("\n" + "─" * 72)
        print("  WRONG MATCHES:")
        for r in wrong_rows:
            print(f"    ✗ {r['img'][:50]}")
            print(f"      predicted : {r['predicted']}")
            print(f"      expected  : {r['ground_truth']}")
            print(f"      vision    : {r['vision_score']:.0%}")

    # ── errors detail ─────────────────────────────────────────────────────────
    if errors:
        print("\n" + "─" * 72)
        print("  ERRORS:")
        for e in errors:
            print(f"    ✗ {e['img']}: {e['error']}")

    # ── per-image table ───────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print(f"  {'#':<3} {'Result':<3} {'Image':<40} {'Predicted':<28} {'GT'}")
    print("─" * 72)
    for i, r in enumerate(rows, 1):
        tick = "✓" if r["correct"] is True else ("✗" if r["correct"] is False else "?")
        gt_short = r["ground_truth"][:25] if r["ground_truth"] else "?"
        pred_short = r["predicted"][:26]
        name_short = r["img"][:38]
        print(f"  {i:<3} {tick:<3} {name_short:<40} {pred_short:<28} {gt_short}")

    # ── optional CSV output ───────────────────────────────────────────────────
    if args.out:
        import csv
        out_path = Path(args.out)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Results written to: {out_path}")

    print("\n" + "═" * 72)
    if rows_with_gt:
        acc = len(correct_rows) / len(rows_with_gt) * 100
        print(f"  REAL ACCURACY: {len(correct_rows)}/{len(rows_with_gt)} = {acc:.1f}%")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
