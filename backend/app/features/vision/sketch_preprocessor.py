"""
OpenCV preprocessing pipeline — morphological background subtraction.

Works for any ink color (blue, black, red, pencil, sketch pen) on any paper
(ruled, grid, plain) under any lighting, because it never looks at color.

Core idea
─────────
  background = morphological CLOSE of the grayscale image with a large kernel.
  The large kernel "paints over" thin strokes while keeping the slow-varying
  background (paper colour + ruled lines + lighting gradient).

  diff = background − original
  Wherever a stroke exists the image is locally darker → diff is large there.
  Wherever the background (ruled lines, paper) exists → diff ≈ 0.

  Threshold diff → clean binary mask of only the strokes, ink-colour-agnostic.

Pipeline
────────
  1. Grayscale
  2. Morphological CLOSE  → background estimate
  3. background − gray    → foreground (stroke) map
  4. Otsu threshold       → binary stroke mask
  5. Connected-component filter → remove noise specks
  6. Crop to content + square white canvas

Falls back to the original image if the result is empty.
Original is never mutated — Ollama still receives the raw upload.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Minimum fraction of pixels that must be ink to trust the result
_MIN_CONTENT_RATIO = 0.002
# Smallest connected component (in pixels) considered a real stroke vs noise
_MIN_BLOB_PX = 40


def preprocess_sketch(image_path: Path) -> Image.Image:
    """
    Extract drawing strokes from a photo of a handwritten sketch.
    Returns a PIL Image (white background, dark strokes) for EfficientNet.
    Falls back to the original on failure or empty result.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        logger.warning("OpenCV could not read %s — using original", image_path)
        return Image.open(image_path).convert("RGB")

    original_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if _is_clean_digital(gray):
        # CAD / clean PNG — return original unchanged so EfficientNet sees exactly
        # what it was trained on.
        logger.debug("%s: clean digital image — no preprocessing", image_path.name)
        return original_pil

    try:
        result = _pipeline(bgr)
    except Exception:
        logger.exception("Preprocessing failed for %s — using original", image_path.name)
        return original_pil

    dark = int(np.sum(result < 200))
    if dark / result.size < _MIN_CONTENT_RATIO:
        logger.debug("%s: preprocessing produced empty result — using original", image_path.name)
        return original_pil

    logger.debug("%s: preprocessed OK (%.2f%% ink pixels)", image_path.name, 100 * dark / result.size)
    return Image.fromarray(result).convert("RGB")


def save_preprocessed(image_path: Path, output_path: Path) -> Path:
    """Preprocess and save — useful for debugging."""
    img = preprocess_sketch(image_path)
    img.save(output_path)
    return output_path


# ── Pipeline ──────────────────────────────────────────────────────────────────

def _pipeline(bgr: np.ndarray) -> np.ndarray:
    """Real-photo path only — clean digital images are handled in preprocess_sketch."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ── 1. Background estimation via morphological CLOSE ─────────────────────
    # Large elliptical kernel "paints over" thin strokes, preserving the slowly-
    # varying background (paper, ruled lines, lighting gradient).
    h, w = gray.shape
    k = max(int(min(h, w) * 0.06) | 1, 21)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # ── 2. Foreground = background − image ───────────────────────────────────
    # Strokes are locally darker → large positive diff.
    # Ruled lines, paper texture → diff ≈ 0 (they ARE the background).
    diff = cv2.subtract(background, gray)

    # ── 3. Otsu threshold on the diff map ────────────────────────────────────
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── 4. Reconnect broken strokes ───────────────────────────────────────────
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)

    # ── 5. Remove noise specks ────────────────────────────────────────────────
    binary = _remove_small_blobs(binary, _MIN_BLOB_PX)

    # ── 6. Measure stroke thickness to decide post-processing strategy ────────
    ink = cv2.bitwise_not(binary)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    max_stroke_radius = float(dist.max())
    thick_marker = max_stroke_radius > 4  # thick marker vs pencil/thin pen

    # ── 7. Strip dimension numbers — only safe for thin strokes ──────────────
    # For thick markers, corner junctions look identical to text blobs (compact
    # bounding box, smallish area) and get incorrectly deleted, destroying the
    # profile shape. Skip this step for thick-stroke images.
    if not thick_marker:
        binary = _strip_text_blobs(binary)

    # ── 8. Thin strokes to ~2-3px ────────────────────────────────────────────
    binary = _thin_strokes(binary, max_stroke_radius)

    # ── 9. Invert to white background / dark strokes (matches training images) ─
    binary = cv2.bitwise_not(binary)

    # ── 10. Isolate the profile: drop floating dimension numbers / annotations ─
    # The single biggest source of noise on real client sketches is the
    # handwritten dimension text ("90", "260", "500", …) sitting beside the
    # profile. It is what the classifier never learned to ignore. Numbers are
    # small connected components separate from the long profile polyline, so we
    # bridge stroke gaps (making the profile one component) and then remove every
    # component that is small relative to the drawing. Large structure — the
    # profile and even a stray table line — is always kept, so we never risk
    # deleting the shape itself.
    binary = _isolate_profile(binary)

    # ── 11. Crop to content ───────────────────────────────────────────────────
    binary = _crop_to_content(binary)

    return binary


def _is_clean_digital(gray: np.ndarray) -> bool:
    """
    True if this looks like a clean digital/CAD image rather than a real photo.
    Heuristic: ≥50% of pixels are pure white (255), which is typical of CAD
    drawings on white background but never true of a phone photo of paper.
    """
    return float((gray == 255).mean()) >= 0.50


# ── Helpers ───────────────────────────────────────────────────────────────────

def _remove_small_blobs(binary: np.ndarray, min_px: int) -> np.ndarray:
    """Remove dark connected components smaller than min_px pixels."""
    inv = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep = np.zeros_like(inv)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_px:
            keep[labels == lbl] = 255
    return cv2.bitwise_not(keep)


def _thin_strokes(binary: np.ndarray, max_stroke_radius: float = 0.0) -> np.ndarray:
    """
    Reduce thick strokes toward ~2-3px using controlled erosion.

    Uses the pre-measured stroke radius so we erode only as much as needed —
    thin pencil strokes get light thinning, thick marker strokes get enough
    erosion to reach ~3px without fragmenting corners and junctions.
    """
    ink = cv2.bitwise_not(binary)
    if ink.sum() == 0:
        return binary

    target_radius = 2.0
    # How many erosion steps to go from current radius to target
    steps = max(0, int(round(max_stroke_radius - target_radius)))
    # Cap at 6 to avoid destroying short segments; min 1 for any real photo
    steps = min(steps, 6)
    if steps == 0:
        return binary

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(ink, k, iterations=steps)

    # If erosion wiped out too much ink, dial back by 1 step
    if eroded.sum() / 255 < ink.sum() / 255 * 0.15:
        eroded = cv2.erode(ink, k, iterations=max(steps - 1, 1))

    if eroded.sum() == 0:
        return binary

    # Dilate back 1px to reconnect any broken stroke tips
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    eroded = cv2.dilate(eroded, k2, iterations=1)
    return cv2.bitwise_not(eroded)


def _strip_text_blobs(binary: np.ndarray,
                      min_aspect: float = 2.5,
                      large_area_px: int = 800) -> np.ndarray:
    """
    Remove dimension numbers/text from a binary sketch image.

    Strategy: connected components whose bounding box is roughly square
    (aspect ratio < min_aspect) AND that are not large enough to be a line
    corner or junction (area < large_area_px) are treated as text and erased.

    Line segments are elongated → high aspect ratio → kept.
    Digits like '4', '00', '150' → compact bounding box → removed.
    """
    inv = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep = np.zeros_like(inv)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        longest = max(bw, bh, 1)
        shortest = max(min(bw, bh), 1)
        aspect = longest / shortest
        if aspect >= min_aspect or area >= large_area_px:
            keep[labels == lbl] = 255
    return cv2.bitwise_not(keep)


def _isolate_profile(
    binary: np.ndarray,
    bridge_iters: int = 2,
    min_diag_frac: float = 0.22,
) -> np.ndarray:
    """
    Keep the profile line-work, drop floating dimension numbers / small text.

    Input/output: white background (255), dark strokes (0).

    Method:
      1. Dilate the ink so a slightly-broken profile becomes ONE component
         (numbers stay separate — they don't touch the profile line).
      2. Measure each component's bounding-box diagonal.
      3. Keep every component whose diagonal is a meaningful fraction of the
         drawing's own diagonal; erase the small ones (digits, arrowheads,
         "c/B" notes). Original stroke thickness is preserved for what we keep.

    Deliberately conservative: it removes only clearly-small blobs, so a large
    table line or a fragmented profile piece survives rather than risk deleting
    the shape. It is a no-op when there is only one component.
    """
    ink = cv2.bitwise_not(binary)
    if int(ink.sum()) == 0:
        return binary

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bridged = cv2.dilate(ink, k, iterations=bridge_iters)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    if num <= 2:  # background + a single component — nothing to isolate
        return binary

    h, w = binary.shape
    img_diag = float(np.hypot(h, w))
    diag_thresh = min_diag_frac * img_diag

    keep_labels = []
    for lbl in range(1, num):
        bw = stats[lbl, cv2.CC_STAT_WIDTH]
        bh = stats[lbl, cv2.CC_STAT_HEIGHT]
        if float(np.hypot(bw, bh)) >= diag_thresh:
            keep_labels.append(lbl)

    if not keep_labels:
        # Everything is "small" (e.g. one tiny sketch) — keep the largest so we
        # never return an empty image.
        largest = max(range(1, num), key=lambda l: stats[l, cv2.CC_STAT_AREA])
        keep_labels = [largest]

    keep_mask = np.isin(labels, keep_labels)
    kept_ink = np.where(keep_mask, ink, 0).astype(np.uint8)
    return cv2.bitwise_not(kept_ink)


def _crop_to_content(binary: np.ndarray, pad_frac: float = 0.08) -> np.ndarray:
    """Crop to ink bounding box, pad, place on square white canvas."""
    ink = cv2.bitwise_not(binary)
    coords = cv2.findNonZero(ink)
    if coords is None:
        return binary

    x, y, bw, bh = cv2.boundingRect(coords)
    px = max(int(bw * pad_frac), 10)
    py = max(int(bh * pad_frac), 10)
    H, W = binary.shape
    x1, y1 = max(x - px, 0), max(y - py, 0)
    x2, y2 = min(x + bw + px, W), min(y + bh + py, H)
    cropped = binary[y1:y2, x1:x2]

    size = max(cropped.shape[0], cropped.shape[1], 1)
    canvas = np.full((size, size), 255, dtype=np.uint8)
    cy = (size - cropped.shape[0]) // 2
    cx = (size - cropped.shape[1]) // 2
    canvas[cy:cy + cropped.shape[0], cx:cx + cropped.shape[1]] = cropped
    return canvas
