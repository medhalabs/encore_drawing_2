"""
Drawing-box auto-detector for the human-in-the-loop batch flow.

Renders each page of a PDF/image and detects candidate bounding boxes around
individual drawings, so the UI can overlay them for the user to adjust before
classification.

Detection strategy (tuned for boxed order forms like Format 1 — Reece):
  - find long horizontal & vertical ruled lines via morphology
  - treat their coordinates (plus the page edges) as a grid
  - each grid cell above a min size is a candidate box
  - keep the tall cells (the drawing areas), drop short field/header rows

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


def detect_boxes(
    page_img: Image.Image,
    min_h_frac: float = 0.13,
    min_w_frac: float = 0.15,
    max_h_frac: float = 0.85,
) -> list[dict]:
    """Return candidate drawing boxes [{x,y,w,h}] for one page."""
    gray = np.array(page_img.convert("L"))
    H, W = gray.shape
    _, binv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, W // 20), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, H // 20)))
    h_lines = cv2.morphologyEx(binv, cv2.MORPH_OPEN, hk)
    v_lines = cv2.morphologyEx(binv, cv2.MORPH_OPEN, vk)

    h_present = (h_lines.sum(axis=1) / 255) > 0.4 * W
    v_present = (v_lines.sum(axis=0) / 255) > 0.4 * H
    ys = _merge_close([0] + _line_centers(h_present) + [H - 1], tol=int(0.03 * H))
    xs = _merge_close([0] + _line_centers(v_present) + [W - 1], tol=int(0.03 * W))

    boxes: list[dict] = []
    for yi in range(len(ys) - 1):
        for xi in range(len(xs) - 1):
            x, y = xs[xi], ys[yi]
            w, h = xs[xi + 1] - x, ys[yi + 1] - y
            if w < min_w_frac * W or h < min_h_frac * H or h > max_h_frac * H:
                continue
            boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

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
