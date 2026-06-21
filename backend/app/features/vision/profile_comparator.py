from pathlib import Path

from app.core.models.schemas import CompareResult
from app.features.masters.loader import MasterRecord
from app.features.ollama.client import OllamaService

COMPARE_PROMPT = """You are a strict engineering drawing matcher for metal roofing flashing profiles.

Sketch (first image): handwritten client drawing
Master (second image): precise CAD machine drawing — {master_key} ({category}, {segment_count} segments)

Think step by step:
1. SKETCH SHAPE: Count the bends/segments. Describe topology (e.g. Z-shape, U-channel, L-angle, hat/capping, step, open apron). Note which direction the profile faces (left/right/up/down) and leg lengths.
2. MASTER SHAPE: Same analysis on the master drawing.
3. ORIENTATION CHECK: Is the sketch a MIRROR IMAGE or 180° ROTATION of the master? If so, treat as DIFFERENT — roofing profiles are direction-specific and a mirrored version is a different product.
4. COMPARE — strict rules:
   - Same segment count = necessary but NOT sufficient
   - Mirrored or flipped = DIFFERENT (score ≤ 0.30)
   - Different topology (open vs closed, U vs Z) = DIFFERENT
   - A Gutter (U-channel open top) and a Capping (closed hat) are NEVER the same
   - An Apron (flat step) and a Capping are NEVER the same
   - Missing or extra legs = DIFFERENT even if overall count is close
   - Only score ≥ 0.70 if you are confident the shapes are the same product

Return ONLY valid JSON:
{{"sketch_shape": "one phrase", "master_shape": "one phrase", "same_topology": true/false, "is_mirrored": true/false, "score": 0.0-1.0, "reasoning": "one sentence"}}

Score guide: 0.85-1.0=same topology same orientation, 0.70-0.84=same topology minor proportion difference, 0.40-0.69=uncertain, 0.0-0.39=different topology or mirrored"""

FEEDBACK_COMPARE_PROMPT = """Compare two handwritten/reference sketch images of metal flashing profiles.

Are these the SAME profile template (same bends and shape topology)? Dimensions may differ.

Return ONLY JSON:
{{"score": 0.0-1.0, "reasoning": "brief"}}"""


class ProfileComparator:
    def __init__(self, ollama: OllamaService):
        self.ollama = ollama

    def compare(self, sketch_path: Path, master: MasterRecord) -> CompareResult:
        prompt = COMPARE_PROMPT.format(
            master_key=master.key,
            category=master.category,
            segment_count=master.segment_count,
        )
        data = self.ollama.chat_vision_json(
            prompt,
            [sketch_path, master.image_path],
            system=(
                "You are a strict engineering drawing matcher. "
                "Penalize different profile topologies heavily. "
                "Focus on shape topology, not dimensions."
            ),
        )
        score = float(data.get("score", 0.0))
        if data.get("same_topology") is False:
            score = min(score, 0.35)
        if data.get("is_mirrored") is True:
            score = min(score, 0.30)
        reasoning = (
            f"sketch={data.get('sketch_shape','?')} master={data.get('master_shape','?')} "
            f"same_topology={data.get('same_topology')} mirrored={data.get('is_mirrored')} {data.get('reasoning','')}"
        )
        return CompareResult(
            master_key=master.key,
            score=score,
            reasoning=reasoning,
        )

    def compare_to_reference_image(self, sketch_path: Path, reference_path: Path) -> CompareResult:
        data = self.ollama.chat_vision_json(
            FEEDBACK_COMPARE_PROMPT,
            [sketch_path, reference_path],
            system="You compare sketch images for profile template similarity.",
        )
        return CompareResult(
            master_key=reference_path.stem,
            score=float(data.get("score", 0.0)),
            reasoning=str(data.get("reasoning", "")),
        )
