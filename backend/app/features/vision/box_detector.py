"""
Drawing-box auto-detector for the human-in-the-loop batch flow.

Renders each page of a PDF/image and detects candidate bounding boxes around
individual drawings, so the UI can overlay them for the user to adjust before
classification.

Detection strategy (works across order-form layouts, e.g. Format 1 — Reece,
Format 2 — flashit.app/ProFinish):
  - find long horizontal ruled lines that span most of the page width; these
    reliably mark row boundaries (table row separators, section dividers) in
    every layout we've seen, so a page-wide threshold is safe here
  - within each resulting horizontal band, separately look for vertical
    dividers local to THAT band (spanning most of the band's own height) —
    some formats only draw column dividers within one row block rather than
    down the whole page (Format 2), so a page-wide vertical threshold misses
    them; scoping the search to the band fixes that without a global rescan
  - keep the resulting cells above a min size (the drawing areas), drop short
    field/header rows

The user can add / move / resize / delete boxes afterwards, so the detector
only needs to get close.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

RENDER_DPI = 150


@dataclass
class PageBoxes:
    index: int
    image: Image.Image          # rendered RGB page
    width: int
    height: int
    boxes: list[dict]           # [{x, y, w, h}] in page-pixel coords


def render_pages(file_path: Path) -> list[Image.Image]:
    """Rasterise a PDF (all pages) or load an image, as RGB PIL images."""
    if file_path.suffix.lower() == ".pdf":
        import fitz  # pymupdf
        doc = fitz.open(str(file_path))
        pages = []
        for page in doc:
            mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
            pix = page.get_pixmap(matrix=mat)
            pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        return pages
    return [Image.open(file_path).convert("RGB")]


def _line_centers(present: np.ndarray) -> list[int]:
    """Collapse runs of True into their center index."""
    out, in_run, start = [], False, 0
    for i, v in enumerate(present):
        if v and not in_run:
            in_run, start = True, i
        elif not v and in_run:
            in_run = False
            out.append((start + i) // 2)
    if in_run:
        out.append((start + len(present)) // 2)
    return out


def _merge_close(vals: list[int], tol: int) -> list[int]:
    out: list[int] = []
    for v in sorted(vals):
        if out and v - out[-1] < tol:
            continue
        out.append(v)
    return out


def _find_column_gap(
    binv_band: np.ndarray, W: int, search_lo: float = 0.1, search_hi: float = 0.85, min_gap_frac: float = 0.015
) -> int | None:
    """Find the widest whitespace gutter between side-by-side cells in a band
    that has no full-height ruled divider (e.g. Format 2's per-card grid boxes,
    separated by a gap rather than a shared column rule). Uses a percentile
    cutoff (not a fixed density) so it adapts to how busy each band's own
    content is, rather than assuming near-zero ink at the gap."""
    density = binv_band.sum(axis=0) / 255.0
    k = max(3, W // 150)
    smooth = np.convolve(density, np.ones(k) / k, mode="same")
    lo, hi = int(search_lo * W), int(search_hi * W)
    seg = smooth[lo:hi]
    if len(seg) == 0:
        return None

    # A real whitespace gutter is close to empty, not just "less busy than
    # 75% of the band". Photo scans never truly hit near-zero (paper texture,
    # shadow gradients keep every column non-trivially inked), so requiring
    # the gap's own minimum to sit near zero — not merely below a percentile —
    # keeps this from firing spuriously on noisy scans.
    if seg.min() > 0.15 * seg.max():
        return None
    is_gap = seg <= max(float(np.percentile(seg, 25)), 1.0)

    best_start, best_len, run_start = 0, 0, None
    for i, gap in enumerate(is_gap.tolist() + [False]):
        if gap and run_start is None:
            run_start = i
        elif not gap and run_start is not None:
            if i - run_start > best_len:
                best_start, best_len = run_start, i - run_start
            run_start = None

    if best_len < min_gap_frac * W:
        return None
    return lo + best_start + best_len // 2


def _extend_with_header_rows(
    all_bands: list[tuple[int, int]], content_idx: int, min_h_frac: float, H: int
) -> int:
    """Extend a content band's top edge upward to absorb the field-label rows
    immediately above it (e.g. NAME/MATERIAL/QTY/DESC on Format 1) — those
    describe the drawing right below them, so the reviewable box should
    include them rather than just the blank sketch area. Stops at the first
    preceding band that isn't header-row-sized: a run of near-uniform small
    rows is what distinguishes NAME/MATERIAL/QTY/DESC from a much taller,
    unrelated block further up the page (e.g. customer/job info fields)."""
    y0 = all_bands[content_idx][0]
    ref_h: int | None = None
    for i in range(content_idx - 1, -1, -1):
        by0, by1 = all_bands[i]
        bh = by1 - by0
        if bh >= min_h_frac * H:
            break
        if ref_h is not None and not (0.4 * ref_h <= bh <= 2.5 * ref_h):
            break
        y0 = by0
        ref_h = bh
    return y0


def _trim_leading_notes(
    y0: int, y1: int, h_lines_loose: np.ndarray, W: int, min_gap_ratio: float = 2.5
) -> int | None:
    """If a band's front portion is unrelated notes/contact-info text with no
    ruled line separating it from the real card table below (e.g. Format 2's
    "Purchasing Team" block sharing a band with row 1's cards), find where the
    real table starts and trim to it. Uses a much looser ink threshold scoped
    to just this band (real dividers there are often light gray, and scoping
    it avoids the grid-row fragmentation a page-wide loose search would cause
    elsewhere). Returns None when the band already starts at its own content
    (no large leading gap to trim)."""
    sub = h_lines_loose[y0:y1, :]
    present = (sub.sum(axis=1) / 255) > 0.4 * W
    centers = _line_centers(present)
    if len(centers) < 2:
        return None
    first_gap = centers[0]
    row_gaps = [b - a for a, b in zip(centers, centers[1:])]
    median_gap = float(np.median(row_gaps))
    if median_gap <= 0 or first_gap < min_gap_ratio * median_gap:
        return None
    return y0 + centers[0]


def detect_boxes(
    page_img: Image.Image,
    min_h_frac: float = 0.13,
    min_w_frac: float = 0.15,
    max_h_frac: float = 0.85,
) -> list[dict]:
    """Return candidate drawing boxes [{x,y,w,h}] for one page."""
    gray = np.array(page_img.convert("L"))
    H, W = gray.shape
    # Row lines: strict cutoff. A looser one also lights up individual grid-paper
    # rows inside a drawing (Format 2's graph-paper background), fragmenting what
    # should be one drawing band into dozens of useless grid-row-sized slivers.
    _, binv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Column dividers: some layouts (Format 2) draw them in light gray — a strict
    # cutoff misses them entirely — but this mask is only ever searched within a
    # single row-band for a near-full-height run, so it doesn't inherit the
    # grid-row fragmentation problem the row search would have.
    _, binv_cols = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, W // 20), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, H // 20)))
    h_lines = cv2.morphologyEx(binv, cv2.MORPH_OPEN, hk)
    v_lines = cv2.morphologyEx(binv_cols, cv2.MORPH_OPEN, vk)
    h_lines_loose = cv2.morphologyEx(binv_cols, cv2.MORPH_OPEN, hk)

    # Row bands: horizontal lines that span most of the page width mark row
    # boundaries reliably in every layout seen so far.
    h_present = (h_lines.sum(axis=1) / 255) > 0.4 * W
    ys = _merge_close([0] + _line_centers(h_present) + [H - 1], tol=int(0.02 * H))

    def find_local_divider(y0: int, y1: int) -> list[int]:
        # High bar (not just >half): a drawing's own tall dimension strokes can
        # reach ~50% of the band height and would otherwise be mistaken for a
        # column rule, which spans effectively the full band top-to-bottom.
        band_v = v_lines[y0:y1, :]
        v_present = (band_v.sum(axis=0) / 255) > 0.85 * (y1 - y0)
        return _line_centers(v_present)

    all_bands = [(ys[yi], ys[yi + 1]) for yi in range(len(ys) - 1)]

    # (drawing_y0, drawing_y1, box_y0) — box_y0 may sit above drawing_y0 once
    # preceding header rows are absorbed; divider search still uses the
    # drawing's own sub-range, since header rows are too short to reliably
    # carry a full-height rule.
    bands = []
    for idx, (y0, y1) in enumerate(all_bands):
        band_h = y1 - y0
        if band_h < min_h_frac * H or band_h > max_h_frac * H:
            continue
        trimmed = _trim_leading_notes(y0, y1, h_lines_loose, W)
        box_y0 = trimmed if trimmed is not None else _extend_with_header_rows(all_bands, idx, min_h_frac, H)
        bands.append((y0, y1, box_y0))

    # A band that also swallows unrelated content above the real table (e.g. a
    # notes section with no ruled line separating it from row 1) dilutes a real
    # divider's coverage below the confidence bar. Some *other* band on the same
    # page usually still finds it cleanly — reuse that x as the page's column
    # template rather than re-deriving it per band.
    template_split: list[int] | None = None
    for y0, y1, _ in bands:
        local = find_local_divider(y0, y1)
        if local:
            template_split = local
            break

    boxes: list[dict] = []
    for y0, y1, box_y0 in bands:
        split_x = find_local_divider(y0, y1) or template_split
        if not split_x:
            gap = _find_column_gap(binv[y0:y1, :], W)
            if gap is not None:
                split_x = [gap]
        xs = _merge_close([0] + (split_x or []) + [W - 1], tol=int(0.02 * W))

        for xi in range(len(xs) - 1):
            x0, x1 = xs[xi], xs[xi + 1]
            w = x1 - x0
            if w < min_w_frac * W:
                continue
            boxes.append({"x": int(x0), "y": int(box_y0), "w": int(w), "h": int(y1 - box_y0)})

    boxes.sort(key=lambda b: (b["y"], b["x"]))
    return boxes


def detect_file(file_path: Path) -> list[PageBoxes]:
    pages = render_pages(file_path)
    out: list[PageBoxes] = []
    for i, img in enumerate(pages):
        boxes = detect_boxes(img)
        out.append(PageBoxes(index=i, image=img, width=img.width, height=img.height, boxes=boxes))
        logger.info("box_detector: page %d -> %d boxes", i, len(boxes))
    return out
