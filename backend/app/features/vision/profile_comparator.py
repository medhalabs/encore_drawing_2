from pathlib import Path

from app.core.models.schemas import CompareResult
from app.features.masters.loader import MasterRecord
from app.features.ollama.client import OllamaService

COMPARE_PROMPT = """You are a strict engineering drawing matcher for metal roofing flashing profiles.

Sketch (first image): handwritten client drawing
Master (second image): precise CAD machine drawing — {master_key} ({category}, {segment_count} segments)

Think step by step:
1. SKETCH SHAPE: Count the bends/segments. Describe the overall topology (e.g. Z-shape, U-channel, L-angle, hat/capping, step, open apron). Note left/right direction.
2. MASTER SHAPE: Same analysis on the master drawing.
3. COMPARE: Are these the SAME template? Rules:
   - Same number of bends = necessary condition
   - Rotated or mirrored (flipped) versions of the same shape = SAME
   - Different topology (e.g. open vs closed, U-channel vs Z-shape) = DIFFERENT even if segment count matches
   - A Gutter (U-channel open at top) and a Capping (closed hat shape) are NEVER the same even with equal segments
   - Small handwriting differences in dimensions are irrelevant — topology only

Return ONLY valid JSON:
{{"sketch_shape": "one phrase", "master_shape": "one phrase", "same_topology": true/false, "score": 0.0-1.0, "reasoning": "one sentence"}}

Score guide: 1.0=identical topology, 0.75-0.95=same topology different proportions, 0.4-0.6=uncertain, 0.0-0.35=different topology"""

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
        reasoning = (
            f"sketch={data.get('sketch_shape','?')} master={data.get('master_shape','?')} "
            f"same_topology={data.get('same_topology')} {data.get('reasoning','')}"
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
