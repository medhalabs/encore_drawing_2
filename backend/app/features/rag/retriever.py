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
        self._vector_scores: dict[str, float] = {}
        self._wrong_masters: set[str] = set()

    def set_feedback_entries(self, entries: list[FeedbackEntry]) -> None:
        self._feedback_entries = entries
        self._wrong_masters = {
            e.previous_master_key for e in entries if e.previous_master_key
        }

    def set_image_boosts(self, boosts: dict[str, float]) -> None:
        self._image_boosts = boosts

    def set_vector_scores(self, scores: dict[str, float]) -> None:
        self._vector_scores = scores

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
            # Use flip-invariant angle distance via fingerprint
            master = self.catalog.get_by_key(master_key)
            if master:
                angle_dist = master.fingerprint.angle_distance(analysis.angles_estimate)
                if angle_dist < 25:
                    boost = max(boost, self.FEEDBACK_BOOST * 0.8)
                    reasons.append(f"feedback_angle_match:{entry.feedback_id[:8]}")
        return boost, reasons

    def _fingerprint_score(self, master: MasterRecord, analysis: SketchAnalysis) -> tuple[float, list[str]]:
        """Deterministic score using exact master geometry — never relies on LLM angle guessing."""
        fp = master.fingerprint
        score = 0.0
        reasons: list[str] = []

        # Angle distance (flip-invariant)
        if analysis.angles_estimate:
            angle_dist = fp.angle_distance(analysis.angles_estimate)
            if angle_dist < 999:
                angle_score = max(0.0, 30.0 - angle_dist)
                score += angle_score
                if angle_dist < 15:
                    reasons.append(f"fp_angle_dist={angle_dist:.1f}")
                elif angle_dist < 30:
                    reasons.append(f"fp_angle_close={angle_dist:.1f}")

        # Length ratio distance (scale-invariant)
        if analysis.handwritten_lengths:
            ratio_dist = fp.length_ratio_distance(analysis.handwritten_lengths)
            if ratio_dist < 999:
                ratio_score = max(0.0, 20.0 - ratio_dist * 40)
                score += ratio_score
                if ratio_dist < 0.15:
                    reasons.append(f"fp_ratio_dist={ratio_dist:.2f}")

        # Fold presence match
        has_fold_hint = bool(analysis.fold_hints and analysis.fold_hints.strip())
        master_has_fold = fp.has_start_fold or fp.has_end_fold
        if has_fold_hint == master_has_fold:
            score += 5.0
            if has_fold_hint:
                reasons.append("fp_fold_match")

        return score, reasons

    def retrieve(self, analysis: SketchAnalysis, top_k: int = 10) -> list[RetrievalCandidate]:
        from app.config.settings import get_settings
        settings = get_settings()

        masters = self.catalog.masters

        # Hard part-class filter when analyze confidence is high — shrinks collision space
        if analysis.confidence >= 0.80 and analysis.part_class_hint:
            filtered = [
                m for m in masters
                if self._part_class_match(analysis.part_class_hint, m.drawing.part_class) > 0.5
            ]
            if len(filtered) >= 3:
                masters = filtered

        candidates: list[RetrievalCandidate] = []

        for master in masters:
            reasons: list[str] = []
            base_score = 0.0

            # Segment count
            seg_diff = abs(master.segment_count - analysis.segment_count)
            if seg_diff == 0:
                base_score += 40
                reasons.append("segment_count_match")
            elif seg_diff == 1:
                base_score += 10
                reasons.append("segment_count_close")
            else:
                base_score -= seg_diff * 15

            # Part class (soft signal)
            part_score = self._part_class_match(analysis.part_class_hint, master.drawing.part_class)
            base_score += part_score * 15
            if part_score > 0:
                reasons.append("part_class_match")

            # Deterministic fingerprint score (geometry-exact)
            fp_score, fp_reasons = self._fingerprint_score(master, analysis)
            base_score += fp_score
            reasons.extend(fp_reasons)

            # Vector similarity
            vector_sim = self._vector_scores.get(master.key, 0.0)
            if self._vector_scores:
                score = (
                    settings.retrieval_vector_weight * vector_sim * 100
                    + settings.retrieval_rule_weight * base_score
                )
                if vector_sim > 0:
                    reasons.append(f"vector_sim={vector_sim:.2f}")
            else:
                score = base_score

            # Feedback boosts
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
