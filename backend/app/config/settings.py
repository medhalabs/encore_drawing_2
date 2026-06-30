from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "https://ollama.com"
    ollama_embed_base_url: str = "https://api.ollama.cloud"
    ollama_api_key: str = ""
    ollama_vision_model: str = "gemma4:31b-cloud"        # used for compare (accuracy matters)
    ollama_analyze_model: str = "gemma3:27b-cloud"         # used for analyze (speed matters)
    ollama_llm_text_model: str = "gpt-oss:120b-cloud"
    ollama_embed_model: str = "nomic-embed-text"

    retrieval_vector_weight: float = 0.35
    retrieval_rule_weight: float = 0.65
    vector_search_top_k: int = 20
    match_top_k: int = 3
    analyzer_consensus_runs: int = 1

    master_drawings_path: str = "../training_testing_datasets/Training/Encore_master_drawings"
    feedback_dir: str = "../training_testing_datasets/feedback"
    upload_dir: str = "./data/uploads"
    cors_origins: str = "http://localhost:3000"
    max_upload_bytes: int = 10 * 1024 * 1024

    database_url: str = "postgresql+asyncpg://encore:encore@localhost:5455/encore_drawings"
    redis_url: str = "redis://localhost:6377/0"
    redis_cache_ttl_seconds: int = 86400
    log_level: str = "INFO"

    min_vision_score: float = 0.72
    no_match_vision_threshold: float = 0.55
    feedback_image_match_threshold: float = 0.72
    feedback_image_boost: float = 60.0
    wrong_master_penalty: float = 35.0

    @property
    def master_drawings_dir(self) -> Path:
        path = Path(self.master_drawings_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path.resolve()

    @property
    def feedback_path(self) -> Path:
        path = Path(self.feedback_dir)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path.resolve()

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path.resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
