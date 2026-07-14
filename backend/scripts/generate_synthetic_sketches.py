"""
Synthetic hand-drawn sketch generator for flashing profiles.

Reads each master's JSON geometry (lengths, angles, firstSegmentAngle) and
renders unlimited hand-drawn-style training sketches: wobbly strokes, variable
pen width/colour, dimension numbers beside segments, angle arcs, arrows and
"c/B" callouts, stray scribbles, ruled/grid paper, photo shadows and JPEG
artifacts — matching the messy client sketches the classifier sees in
production.

This is the main cure for the classifier's data starvation: ~15 real images
per class plateaued at ~39% top-1 on the client test set; adding synthetic
data lifted it to 48%+ (see scripts/train_from_dl.py to retrain).

Geometry convention (validated against the master PNGs):
  - screen coordinates, y-down; heading starts at firstSegmentAngle degrees
  - each bend turns by angles[i]
  - the "-mirror" variant is a horizontal flip (flip_h)

Usage (from backend/):
    ./.venv/bin/python scripts/generate_synthetic_sketches.py            # 300/class
    ./.venv/bin/python scripts/generate_synthetic_sketches.py --per-class 100
    ./.venv/bin/python scripts/generate_synthetic_sketches.py --out /tmp/synth
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
MASTERS = ROOT / "training_testing_datasets" / "Training" / "Encore_master_drawings"
DEFAULT_OUT = ROOT / "training_testing_datasets" / "Training" / "training_synth"

FONTS = [
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/Chalkboard.ttc",
    "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
    "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf",
]


def load_geometry(json_path: Path) -> tuple[list[float], list[float], float]:
    d = json.loads(json_path.read_text())
    return d["lengths"], d["angles"], d.get("firstSegmentAngle", 90)


def polyline(lengths, angles, first_angle=90.0, flip_h=False):
    """Screen coords (y-down). Heading starts at first_angle; each bend turns angles[i]."""
    pts = [(0.0, 0.0)]
    heading = math.radians(first_angle)
    for i, L in enumerate(lengths):
        x, y = pts[-1]
        pts.append((x + L * math.cos(heading), y + L * math.sin(heading)))
        if i < len(angles):
            heading += math.radians(angles[i])
    if flip_h:
        pts = [(-x, y) for x, y in pts]
    return pts


def fit(pts, size, margin):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    w = max(xs) - min(xs) or 1; h = max(ys) - min(ys) or 1
    s = (size - 2 * margin) / max(w, h)
    ox = (size - s * w) / 2 - s * min(xs)
    oy = (size - s * h) / 2 - s * min(ys)
    return [(ox + s * x, oy + s * y) for x, y in pts], s


def _smooth_noise(n, amp):
    t = np.linspace(0, 1, n)
    out = np.zeros(n)
    for f in (1, 2, 3, 5):
        out += (random.uniform(-1, 1) / f) * np.sin(2 * np.pi * f * t + random.uniform(0, 6.28))
    return out * amp


def wobble_segment(p0, p1, wobble):
    x0, y0 = p0; x1, y1 = p1
    L = math.hypot(x1 - x0, y1 - y0)
    n = max(int(L / 4), 4)
    t = np.linspace(0, 1, n)
    xs = x0 + (x1 - x0) * t
    ys = y0 + (y1 - y0) * t
    px, py = -(y1 - y0) / L, (x1 - x0) / L
    disp = _smooth_noise(n, wobble)
    envelope = np.sin(np.pi * t) ** 0.5
    xs += px * disp * envelope
    ys += py * disp * envelope
    return list(zip(xs, ys))


def draw_stroke(draw, path_pts, width, color):
    for i in range(len(path_pts) - 1):
        draw.line([path_pts[i], path_pts[i + 1]], fill=color, width=width)
    r = width / 2
    for x, y in path_pts:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def _font(size_range=(18, 30)):
    try:
        return ImageFont.truetype(random.choice(FONTS), random.randint(*size_range))
    except Exception:
        return ImageFont.load_default()


def render_sketch(lengths, angles, first_angle, flip_h=False, size=560):
    """One synthetic hand-drawn sketch as a PIL RGB image."""
    paper = random.choice(["white", "white", "offwhite", "ruled", "grid"])
    bg_val = 255 if paper == "white" else random.randint(238, 250)
    img = Image.new("RGB", (size, size), (bg_val, bg_val, max(bg_val - random.randint(0, 6), 230)))
    d = ImageDraw.Draw(img)
    if paper == "ruled":
        gap = random.randint(28, 44); c = random.randint(190, 225)
        for y in range(gap, size, gap):
            d.line([(0, y), (size, y)], fill=(c, c, min(c + 20, 255)), width=1)
    elif paper == "grid":
        gap = random.randint(24, 36); c = random.randint(200, 230)
        for y in range(gap, size, gap):
            d.line([(0, y), (size, y)], fill=(c, c, c), width=1)
        for x in range(gap, size, gap):
            d.line([(x, 0), (x, size)], fill=(c, c, c), width=1)

    # hand drawings are never to scale — jitter lengths/angles per sketch
    jl = [L * random.uniform(0.8, 1.25) for L in lengths]
    ja = [a + random.uniform(-6, 6) for a in angles]
    jfa = first_angle + random.uniform(-6, 6)
    pts = polyline(jl, ja, jfa, flip_h)
    margin = random.randint(70, 110)
    spts, _ = fit(pts, size, margin)

    ang = math.radians(random.uniform(-7, 7))
    cx = cy = size / 2
    spts = [((x - cx) * math.cos(ang) - (y - cy) * math.sin(ang) + cx,
             (x - cx) * math.sin(ang) + (y - cy) * math.cos(ang) + cy) for x, y in spts]

    ink_choice = random.choice(["black", "black", "blue", "dark"])
    ink = {"black": (20, 20, 20), "blue": (30, 45, random.randint(120, 200)),
           "dark": (50, 50, 55)}[ink_choice]
    width = random.choice([2, 2, 3, 3, 4, 5, 7])
    wobble = random.uniform(0.8, 3.2) * (1 + width / 8)

    sketchy = random.random() < 0.25  # some people retrace their lines
    for i in range(len(spts) - 1):
        p0, p1 = spts[i], spts[i + 1]
        ov = random.uniform(-0.03, 0.06)
        p1e = (p1[0] + (p1[0] - p0[0]) * ov, p1[1] + (p1[1] - p0[1]) * ov)
        draw_stroke(d, wobble_segment(p0, p1e, wobble), width, ink)
        if sketchy:
            draw_stroke(d, wobble_segment(p0, p1e, wobble * 1.4), max(width - 1, 1), ink)

    # dimension numbers beside segment midpoints
    # (an experiment placing digits touching the line tested WORSE — 45% vs 52%
    # top-1 on the client eval — keep numbers neatly beside the segments)
    for i in range(len(spts) - 1):
        if random.random() < 0.15:
            continue
        p0, p1 = spts[i], spts[i + 1]
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) or 1
        px, py = -(p1[1] - p0[1]) / L, (p1[0] - p0[0]) / L
        off = random.uniform(16, 34) * random.choice([1, -1])
        label = str(int(lengths[i] * random.choice([1, 1, 1, 10, 0.1]) if random.random() < 0.2
                        else random.choice([10, 20, 30, 40, 50, 60, 75, 90, 100, 130, 150,
                                            200, 245, 260, 300, 430, 500, 600])))
        d.text((mx + px * off, my + py * off), label, fill=ink, font=_font(), anchor="mm")

    # angle arc + degree label at a corner
    if random.random() < 0.35 and len(spts) > 2:
        ci = random.randint(1, len(spts) - 2)
        cxp, cyp = spts[ci]
        r0 = random.uniform(14, 26)
        d.arc([cxp - r0, cyp - r0, cxp + r0, cyp + r0],
              random.randint(0, 180), random.randint(200, 360), fill=ink, width=max(width - 2, 1))
        deg = random.choice([30, 45, 90, 110, 120, 135, 150])
        d.text((cxp + r0 + 6, cyp - r0), f"{deg}°", fill=ink, font=_font((14, 20)))

    # stray scribble away from the profile (clutter robustness)
    if random.random() < 0.3:
        sx = random.uniform(20, size - 60)
        sy = random.choice([random.uniform(10, 50), random.uniform(size - 60, size - 15)])
        pts_s = [(sx + i * random.uniform(4, 9), sy + random.uniform(-6, 6))
                 for i in range(random.randint(4, 9))]
        d.line(pts_s, fill=ink, width=max(width - 2, 1))

    # table/edge lines (client sheets often have forms)
    if random.random() < 0.2:
        ex = random.choice([random.uniform(6, 30), random.uniform(size - 40, size - 8)])
        d.line([(ex, 0), (ex, size)], fill=ink, width=random.randint(1, 3))
    if random.random() < 0.15:
        ey = random.uniform(6, 40)
        d.line([(0, ey), (size, ey)], fill=ink, width=random.randint(1, 3))

    # arrow + note callout ("c/B" etc.)
    if random.random() < 0.5:
        ax, ay = random.uniform(size * 0.55, size * 0.85), random.uniform(size * 0.15, size * 0.45)
        bx, by = ax - random.uniform(30, 70), ay + random.uniform(20, 50)
        d.line([(bx, by), (ax, ay)], fill=ink, width=max(width - 1, 2))
        d.text((ax + 8, ay - 8), random.choice(["c/B", "c/b", "CB", "col", "Z"]),
               fill=ink, font=_font((16, 24)))

    # photo effects: shadow gradient, blur, jpeg artifacts
    if random.random() < 0.35:
        arr = np.asarray(img).astype(np.float32)
        gx = np.linspace(random.uniform(-30, 0), random.uniform(0, 30), size)
        gy = np.linspace(random.uniform(-30, 0), random.uniform(0, 30), size)
        arr += gx[None, :, None] + gy[:, None, None]
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 0.9)))
    if random.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(35, 75))
        img = Image.open(buf).convert("RGB")
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    random.seed(args.seed)
    t0 = time.time()
    count = 0
    for cat_dir in sorted(MASTERS.iterdir()):
        if not cat_dir.is_dir():
            continue
        for jp in sorted(cat_dir.glob("*.json")):
            lengths, angles, fa = load_geometry(jp)
            out_dir = args.out / cat_dir.name / jp.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(args.per_class):
                render_sketch(lengths, angles, fa).save(out_dir / f"synth-{i:03d}.png")
                count += 1
            print(f"{cat_dir.name}/{jp.stem}: {args.per_class} done", flush=True)
    print(f"TOTAL {count} images in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
