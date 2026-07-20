#!/usr/bin/env python3
"""
Creates side-by-side PNG grids: handwritten sketch (left) vs matched master (right).
Outputs one image per pair into /tmp/comparisons/, plus a full summary grid.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

MASTER_ROOT = Path("/Users/pavanrajkg/Documents/Pavan/encore_drawing_2/training_testing_datasets/Training/Encore_master_drawings")
TEST_ROOT   = Path("/Users/pavanrajkg/Documents/Pavan/encore_drawing_2/training_testing_datasets/testing/Client_handwritten_data")
OUT_DIR     = Path("/tmp/comparisons")
OUT_DIR.mkdir(exist_ok=True)

# Eval results from the run
NB = " "  # narrow no-break space used in macOS screenshot filenames

RESULTS = [
    (f"Screenshot 2026-04-16 at 1.24.04{NB}PM.png",    "Gutters/gutter-3"),
    (f"Screenshot 2026-04-16 at 1.24.17{NB}PM.png",    "Gutters/gutter-11"),
    (f"Screenshot 2026-04-16 at 1.24.31{NB}PM.png",    "Capping/capping-8"),
    (f"Screenshot 2026-04-16 at 12.40.15{NB}PM.png",   "Capping/capping-1"),
    (f"Screenshot 2026-04-21 at 1.00.35{NB}PM.png",    "Gutters/gutter-11"),
    (f"Screenshot 2026-04-21 at 2.57.17{NB}PM.png",    "Gutters/gutter-11"),
    (f"Screenshot 2026-04-22 at 1.23.36{NB}AM.png",    "Capping/capping-9"),
    (f"Screenshot 2026-04-22 at 1.34.01{NB}AM.png",    "Capping/capping-8"),
    (f"Screenshot 2026-04-22 at 1.34.15{NB}AM.png",    "Capping/capping-1"),
    (f"Screenshot 2026-04-22 at 1.34.24{NB}AM.png",    "Capping/capping-2"),
    (f"Screenshot 2026-04-22 at 1.34.31{NB}AM.png",    "Capping/capping-4"),
    (f"Screenshot 2026-04-22 at 1.34.36{NB}AM.png",    "Gutters/gutter-4"),
    (f"Screenshot 2026-04-22 at 1.34.43{NB}AM.png",    "Aprons/apron-1"),
    (f"Screenshot 2026-04-22 at 1.34.50{NB}AM.png",    "Aprons/apron-3"),
    (f"Screenshot 2026-04-22 at 1.35.00{NB}AM.png",    "Aprons/apron-6"),
    (f"Screenshot 2026-04-22 at 1.35.10{NB}AM.png",    "Gutters/gutter-4"),
    (f"Screenshot 2026-04-27 at 7.50.29{NB}PM.png",    "Capping/capping-1"),
    (f"Screenshot 2026-05-14 at 2.47.36{NB}PM.png",    "Capping/capping-7"),
    (f"Screenshot 2026-05-21 at 12.36.50{NB}AM.png",   "Aprons/apron-2"),
    (f"Screenshot 2026-05-21 at 12.36.54{NB}AM.png",   "Aprons/apron-2"),
    (f"Screenshot 2026-05-21 at 12.36.57{NB}AM.png",   "Capping/capping-3"),
    (f"Screenshot 2026-05-21 at 12.37.05{NB}AM.png",   "Aprons/apron-8"),
    (f"Screenshot 2026-05-21 at 12.37.10{NB}AM.png",   "Aprons/apron-2"),
    (f"Screenshot 2026-05-21 at 12.37.18{NB}AM.png",   "Capping/capping-8"),
    (f"Screenshot 2026-05-21 at 12.37.28{NB}AM.png",   "Capping/capping-8"),
    (f"Screenshot 2026-05-21 at 12.37.41{NB}AM.png",   "Aprons/apron-3"),
    (f"Screenshot 2026-05-21 at 12.37.54{NB}AM.png",   "Capping/capping-9"),
    (f"Screenshot 2026-05-21 at 12.37.57{NB}AM.png",   "Aprons/apron-2"),
    (f"Screenshot 2026-05-21 at 12.38.01{NB}AM.png",   "RidgeValley/ridgevalley-2"),
    (f"Screenshot 2026-05-21 at 12.38.04{NB}AM.png",   "RidgeValley/ridgevalley-2"),
    (f"Screenshot 2026-05-21 at 12.38.19{NB}AM.png",   "Capping/capping-10"),
    (f"Screenshot 2026-05-21 at 12.38.26{NB}AM.png",   "Capping/capping-1"),
    (f"Screenshot 2026-05-21 at 12.38.30{NB}AM.png",   "Capping/capping-9"),
    (f"Screenshot 2026-05-21 at 12.38.35{NB}AM.png",   "Gutters/gutter-3"),
    (f"Screenshot 2026-05-21 at 12.38.42{NB}AM.png",   "Capping/capping-7"),
    (f"Screenshot 2026-05-21 at 12.38.50{NB}AM.png",   "Capping/capping-4"),
    (f"Screenshot 2026-05-21 at 12.38.56{NB}AM.png",   "Capping/capping-8"),
    (f"Screenshot 2026-05-21 at 12.39.02{NB}AM.png",   "Aprons/apron-2"),
    (f"Screenshot 2026-05-21 at 12.39.09{NB}AM.png",   "Capping/capping-8"),
    (f"Screenshot 2026-05-21 at 12.39.15{NB}AM.png",   "Gutters/gutter-8"),
    (f"Screenshot 2026-05-21 at 12.39.19{NB}AM.png",   "Gutters/gutter-11"),
    (f"Screenshot 2026-05-21 at 12.39.32{NB}AM.png",   "Capping/capping-4"),
    (f"Screenshot 2026-05-25 at 10.34.29{NB}AM.png",   "Gutters/gutter-4"),
    (f"Screenshot 2026-05-25 at 10.34.34{NB}AM.png",   "Capping/capping-10"),
    (f"Screenshot 2026-05-25 at 10.34.37{NB}AM.png",   "Capping/capping-8"),
    (f"Screenshot 2026-05-25 at 10.34.41{NB}AM.png",   "Aprons/apron-8"),
    (f"Screenshot 2026-05-25 at 10.34.44{NB}AM.png",   "Aprons/apron-1"),
    (f"Screenshot 2026-05-25 at 10.34.51{NB}AM.png",   "Capping/capping-8"),
    (f"Screenshot 2026-05-25 at 10.37.04{NB}AM.png",   "Gutters/gutter-4"),
    (f"Screenshot 2026-05-25 at 10.37.08{NB}AM.png",   "Capping/capping-8"),
]

THUMB = (400, 300)
LABEL_H = 36
PAD = 8
PAIR_W = THUMB[0] * 2 + PAD * 3
PAIR_H = THUMB[1] + LABEL_H * 2 + PAD * 2

COLS = 5
ROWS = (len(RESULTS) + COLS - 1) // COLS

GRID_W = PAIR_W * COLS + PAD
GRID_H = PAIR_H * ROWS + PAD

try:
    font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    font_idx   = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
except Exception:
    font_label = ImageFont.load_default()
    font_idx   = font_label

grid = Image.new("RGB", (GRID_W, GRID_H), (240, 240, 240))
draw = ImageDraw.Draw(grid)

for i, (sketch_name, master_key) in enumerate(RESULTS):
    sketch_path = TEST_ROOT / sketch_name
    cat, base   = master_key.split("/")
    master_path = MASTER_ROOT / cat / f"{base}.png"

    # Load images
    try:
        sketch_img = Image.open(sketch_path).convert("RGB")
    except Exception:
        sketch_img = Image.new("RGB", THUMB, (200, 200, 200))

    try:
        master_img = Image.open(master_path).convert("RGB")
    except Exception:
        master_img = Image.new("RGB", THUMB, (200, 200, 200))

    sketch_img.thumbnail(THUMB, Image.LANCZOS)
    master_img.thumbnail(THUMB, Image.LANCZOS)

    # Paste onto white card
    card = Image.new("RGB", (PAIR_W, PAIR_H), (255, 255, 255))
    cd   = ImageDraw.Draw(card)

    # Index label
    cd.text((PAD, PAD), f"#{i+1}", fill=(80, 80, 80), font=font_idx)

    # Sketch
    sx = PAD
    sy = LABEL_H + PAD
    card.paste(sketch_img, (sx + (THUMB[0] - sketch_img.width)//2,
                             sy + (THUMB[1] - sketch_img.height)//2))
    cd.text((sx, sy + THUMB[1] + 2), "SKETCH", fill=(50, 50, 200), font=font_label)

    # Master
    mx = THUMB[0] + PAD * 2
    card.paste(master_img, (mx + (THUMB[0] - master_img.width)//2,
                             sy + (THUMB[1] - master_img.height)//2))
    cd.text((mx, sy + THUMB[1] + 2), f"→ {master_key}", fill=(200, 50, 50), font=font_label)

    # Border
    cd.rectangle([0, 0, PAIR_W-1, PAIR_H-1], outline=(180, 180, 180), width=1)

    # Save individual pair
    card.save(OUT_DIR / f"pair_{i+1:02d}.png")

    # Paste into grid
    col = i % COLS
    row = i // COLS
    gx  = col * PAIR_W + PAD
    gy  = row * PAIR_H + PAD
    grid.paste(card, (gx, gy))
    print(f"  [{i+1:02d}/50] {sketch_name[:40]} → {master_key}")

# Save grid in two halves (top 25, bottom 25) for readable size
half = ROWS // 2
h1 = grid.crop((0, 0, GRID_W, half * PAIR_H + PAD))
h2 = grid.crop((0, half * PAIR_H, GRID_W, GRID_H))
h1.save("/tmp/comparisons/grid_1_25.png")
h2.save("/tmp/comparisons/grid_26_50.png")
grid.save("/tmp/comparisons/grid_all.png")

print(f"\nSaved grids to /tmp/comparisons/")
print(f"  grid_1_25.png   — pairs 1–25")
print(f"  grid_26_50.png  — pairs 26–50")
print(f"  grid_all.png    — all 50")
