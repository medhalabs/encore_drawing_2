"""
Batch drawing splitter.

Takes a PDF or image file and returns a list of cropped PIL Images,
one per individual drawing detected on the page.

Strategy (works for clean-scan style like 110-2):
1. Rasterise each PDF page at 200 DPI → RGB image
2. Convert to grayscale, threshold to binary
3. Compute horizontal projection (dark pixels per row)
4. Find gap rows (whitespace / divider lines) → these are separators
5. Crop between separators, discard tiny regions
6. Also handles single-image uploads (returns [original])
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum height (px at 200 DPI) to be considered a real drawing region
_MIN_REGION_HEIGHT = 120
# Horizontal line must span at least this fraction of the page width to be a separator
_LINE_WIDTH_FRACTION = 0.55
# A row is a "gap" if fewer than this many dark pixels (fallback projection method)
_GAP_DARK_THRESHOLD = 4
# Min consecutive gap rows needed to count as a separator (fallback)
_MIN_GAP_ROWS = 40


def split_file(file_path: Path) -> list[Image.Image]:
    """
    Split a PDF or image into individual drawing crops.
    Returns list of PIL Images (RGB), one per drawing found.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        pages = _rasterise_pdf(file_path)
    else:
        pages = [Image.open(file_path).convert("RGB")]

    crops: list[Image.Image] = []
    for page_img in pages:
        page_crops = _split_page(page_img)
        crops.extend(page_crops)

    if not crops:
        # Fallback: return the whole first page
        if pages:
            crops = [pages[0]]

    logger.info("batch_splitter: found %d drawing regions in %s", len(crops), file_path.name)
    return crops


def _rasterise_pdf(pdf_path: Path) -> list[Image.Image]:
    import fitz  # pymupdf
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages


def _split_page(page_img: Image.Image) -> list[Image.Image]:
    """Detect drawing regions in a single page and return cropped PIL Images."""
    gray = np.array(page_img.convert("L"))
    H, W = gray.shape

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # ── Strategy 1: detect explicit full-width horizontal separator lines ─────
    separator_ys = _find_separator_lines(binary, W)

    if len(separator_ys) >= 1:
        # Use detected lines as region boundaries
        boundaries = [0] + separator_ys + [H]
        regions = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
    else:
        # ── Strategy 2: fallback — large whitespace gaps ───────────────────────
        proj = binary.sum(axis=1) / 255
        is_gap = proj < _GAP_DARK_THRESHOLD
        separators = _find_separator_bands(is_gap, min_width=_MIN_GAP_ROWS)
        regions = _bands_to_regions(separators, H)

    crops = []
    for (y1, y2) in regions:
        if y2 - y1 < _MIN_REGION_HEIGHT:
            continue
        crop = page_img.crop((0, y1, W, y2))
        tight = _tight_crop(crop)
        if tight is not None:
            crops.append(tight)

    return crops


def _find_separator_lines(binary: np.ndarray, W: int) -> list[int]:
    """
    Find y-positions of horizontal lines spanning > _LINE_WIDTH_FRACTION of the page.
    These are the divider lines clients draw between drawings.
    Returns sorted list of y-coordinates (midpoint of each line band).
    """
    # Morphological: isolate long horizontal segments
    k_len = int(W * _LINE_WIDTH_FRACTION)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Horizontal projection of detected lines
    proj = h_lines.sum(axis=1) / 255
    threshold = W * _LINE_WIDTH_FRACTION * 0.5  # at least half-width coverage

    # Find bands where long lines exist
    in_band = False
    start = 0
    ys = []
    for i, val in enumerate(proj):
        if val >= threshold and not in_band:
            in_band = True
            start = i
        elif val < threshold and in_band:
            in_band = False
            ys.append((start + i) // 2)  # midpoint
    return ys


def _find_separator_bands(is_gap: np.ndarray, min_width: int) -> list[tuple[int, int]]:
    """Return (start, end) row index pairs for each gap band wider than min_width."""
    bands = []
    in_gap = False
    start = 0
    for i, g in enumerate(is_gap):
        if g and not in_gap:
            in_gap = True
            start = i
        elif not g and in_gap:
            in_gap = False
            if i - start >= min_width:
                bands.append((start, i))
    if in_gap and len(is_gap) - start >= min_width:
        bands.append((start, len(is_gap)))
    return bands


def _bands_to_regions(separators: list[tuple[int, int]], H: int) -> list[tuple[int, int]]:
    """Convert separator bands into content regions between them."""
    boundaries = [0] + [mid for (s, e) in separators for mid in [(s + e) // 2]] + [H]
    regions = []
    for i in range(len(boundaries) - 1):
        regions.append((boundaries[i], boundaries[i + 1]))
    return regions


def _tight_crop(img: Image.Image, pad: int = 20) -> Image.Image | None:
    """Crop to actual ink content with padding. Returns None if image is blank."""
    gray = np.array(img.convert("L"))
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, bw, bh = cv2.boundingRect(coords)
    H, W = gray.shape
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + bw + pad, W)
    y2 = min(y + bh + pad, H)
    if bw < 30 or bh < 30:
        return None
    return img.crop((x1, y1, x2, y2))
