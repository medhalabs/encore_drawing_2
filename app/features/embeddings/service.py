from app.config.settings import Settings
from app.core.models.schemas import SketchAnalysis
from app.features.masters.catalog import MasterCatalog
from app.features.masters.loader import MasterRecord
from app.features.ollama.client import OllamaService


class EmbeddingService:
    def __init__(self, settings: Settings, ollama: OllamaService, catalog: MasterCatalog):
        self.settings = settings
        self.ollama = ollama
        self.catalog = catalog

    def build_master_embed_text(self, master: MasterRecord) -> str:
        fingerprint = self.catalog.fingerprint(master)
        return (
            f"category={master.category} | display_name={master.display_name} | "
            f"{fingerprint}"
        )

    def build_sketch_embed_text(self, analysis: SketchAnalysis) -> str:
        angles = ",".join(str(a) for a in analysis.angles_estimate)
        return (
            f"part_class={analysis.part_class_hint} | segments={analysis.segment_count} | "
            f"angles=[{angles}] | folds={analysis.fold_hints} | "
            f"description={analysis.description}"
        )

    def embed_text(self, text: str) -> list[float]:
        return self.ollama.embed(text)

    def embed_master(self, master: MasterRecord) -> list[float]:
        return self.embed_text(self.build_master_embed_text(master))

    def embed_sketch_analysis(self, analysis: SketchAnalysis) -> list[float]:
        return self.embed_text(self.build_sketch_embed_text(analysis))
