from dataclasses import dataclass, field

from app.core.models.schemas import FeedbackEntry, SketchAnalysis
from app.features.masters.catalog import MasterCatalog
from app.features.masters.loader import MasterRecord


@dataclass
class RetrievalCandidate:
    master: MasterRecord
    score: float
    reasons: list[str] = field(default_factory=list)


class MasterRetriever:
    FEEDBACK_BOOST = 50.0

    def __init__(self, catalog: MasterCatalog):
        self.catalog = catalog
        self._feedback_entries: list[FeedbackEntry] = []
        self._image_boosts: dict[str, float] = {}
        self._wrong_masters: set[str] = set()

    def set_feedback_entries(self, entries: list[FeedbackEntry]) -> None:
        self._feedback_entries = entries
        self._wrong_masters = {
            e.previous_master_key for e in entries if e.previous_master_key
        }

    def set_image_boosts(self, boosts: dict[str, float]) -> None:
        self._image_boosts = boosts

    @staticmethod
    def _angle_distance(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 999.0
        if len(a) != len(b):
            return 999.0
        return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

    @staticmethod
    def _part_class_match(hint: str, part_class: str) -> float:
        if not hint:
            return 0.0
        hint_l = hint.lower().replace(" ", "").replace("&", "")
        part_l = part_class.lower().replace(" ", "").replace("&", "")
        if hint_l in part_l or part_l in hint_l:
            return 1.0
        aliases = {
            "apron": "aprons",
            "gutter": "gutters",
            "capping": "cappings",
            "ridgevalley": "ridgevalley",
            "soaker": "soakers",
            "footmould": "footmoulds",
        }
        for key, val in aliases.items():
            if key in hint_l and val in part_l:
                return 0.8
        return 0.0

    def _feedback_boost(self, analysis: SketchAnalysis, master_key: str) -> tuple[float, list[str]]:
        boost = 0.0
        reasons: list[str] = []
        for entry in self._feedback_entries:
            if entry.master_key != master_key:
                continue
            if entry.segment_count == analysis.segment_count:
                boost = max(boost, self.FEEDBACK_BOOST)
                reasons.append(f"feedback_segment_match:{entry.feedback_id[:8]}")
            angle_dist = self._angle_distance(entry.angles, analysis.angles_estimate)
            if angle_dist < 25:
                boost = max(boost, self.FEEDBACK_BOOST * 0.8)
                reasons.append(f"feedback_angle_match:{entry.feedback_id[:8]}")
        return boost, reasons

    def retrieve(self, analysis: SketchAnalysis, top_k: int = 5) -> list[RetrievalCandidate]:
        from app.config.settings import get_settings
        settings = get_settings()
        candidates: list[RetrievalCandidate] = []

        for master in self.catalog.masters:
            reasons: list[str] = []
            score = 0.0

            seg_diff = abs(master.segment_count - analysis.segment_count)
            if seg_diff == 0:
                score += 40
                reasons.append("segment_count_match")
            elif seg_diff == 1:
                score += 10
                reasons.append("segment_count_close")
            else:
                score -= seg_diff * 15

            angle_dist = self._angle_distance(master.drawing.angles, analysis.angles_estimate)
            if angle_dist < 999:
                angle_score = max(0, 25 - angle_dist)
                score += angle_score
                if angle_dist < 15:
                    reasons.append(f"angle_distance={angle_dist:.1f}")

            part_score = self._part_class_match(analysis.part_class_hint, master.drawing.part_class)
            score += part_score * 15
            if part_score > 0:
                reasons.append("part_class_match")

            fb_boost, fb_reasons = self._feedback_boost(analysis, master.key)
            score += fb_boost
            reasons.extend(fb_reasons)

            img_boost = self._image_boosts.get(master.key, 0.0)
            if img_boost > 0:
                score += img_boost
                reasons.append(f"feedback_image_boost={img_boost:.0f}")

            if master.key in self._wrong_masters:
                score -= settings.wrong_master_penalty
                reasons.append("previously_corrected_away")

            candidates.append(RetrievalCandidate(master=master, score=score, reasons=reasons))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]
