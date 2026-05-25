from pathlib import Path

from app.core.models.schemas import CompareResult
from app.features.masters.loader import MasterRecord
from app.features.ollama.client import OllamaService

COMPARE_PROMPT = """Compare a handwritten sketch (first image) with a master machine drawing (second image).

Both are metal flashing cross-section profiles. Decide if they are the SAME profile template:
- SAME number of bends/segments
- SAME overall shape topology (e.g. both U-shaped, both L-shaped, both zigzag apron)
- Rotated or mirrored versions of the same template count as SAME

Score LOW if:
- Different topology (e.g. U-shape vs L-shape, open vs closed profile)
- Different number of bends
- Only share part category or similar numbers but different geometry

Return ONLY JSON:
{{"score": 0.0-1.0, "reasoning": "brief explanation", "same_topology": true/false}}

Score guide: 1.0 = identical template, 0.7+ = same topology different dims, 0.4 = uncertain, 0.0-0.3 = different profile."""

FEEDBACK_COMPARE_PROMPT = """Compare two handwritten/reference sketch images of metal flashing profiles.

Are these the SAME profile template (same bends and shape topology)? Dimensions may differ.

Return ONLY JSON:
{{"score": 0.0-1.0, "reasoning": "brief"}}"""


class ProfileComparator:
    def __init__(self, ollama: OllamaService):
        self.ollama = ollama

    def compare(self, sketch_path: Path, master: MasterRecord) -> CompareResult:
        data = self.ollama.chat_vision_json(
            COMPARE_PROMPT,
            [sketch_path, master.image_path],
            system="You are a strict engineering drawing matcher. Penalize different profile topologies heavily.",
        )
        score = float(data.get("score", 0.0))
        if data.get("same_topology") is False:
            score = min(score, 0.35)
        return CompareResult(
            master_key=master.key,
            score=score,
            reasoning=str(data.get("reasoning", "")),
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
