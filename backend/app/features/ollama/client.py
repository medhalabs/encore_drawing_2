import base64
import json
import logging
import re
from pathlib import Path

from ollama import Client

from app.config.settings import Settings
from app.features.cache import redis_cache

logger = logging.getLogger(__name__)


class OllamaService:
    def __init__(self, settings: Settings):
        self.settings = settings
        auth_headers = {"Authorization": f"Bearer {settings.ollama_api_key}"}
        self.client = Client(host=settings.ollama_base_url, headers=auth_headers)
        self.embed_client = Client(host=settings.ollama_embed_base_url, headers=auth_headers)

    @staticmethod
    def _image_to_base64(path: Path) -> str:
        return base64.b64encode(path.read_bytes()).decode("utf-8")

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        return json.loads(text)

    def chat_text(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat(
            model=self.settings.ollama_llm_text_model,
            messages=messages,
            stream=False,
        )
        return response["message"]["content"]

    def chat_vision(self, prompt: str, image_paths: list[Path], system: str = "", model: str = "") -> str:
        images = [self._image_to_base64(p) for p in image_paths]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": images})
        response = self.client.chat(
            model=model or self.settings.ollama_vision_model,
            messages=messages,
            stream=False,
        )
        return response["message"]["content"]

    def chat_vision_json(self, prompt: str, image_paths: list[Path], system: str = "", model: str = "") -> dict:
        resolved_model = model or self.settings.ollama_vision_model
        logger.info(
            "chat_vision_json model=%s images=%s prompt_len=%d",
            resolved_model,
            [p.name for p in image_paths],
            len(prompt),
        )
        cache_key = redis_cache.build_vision_cache_key(prompt, image_paths)
        cached = redis_cache.cache_get_sync(cache_key)
        if cached is not None:
            logger.debug("Redis cache hit for vision JSON (%s)", [p.name for p in image_paths])
            return cached
        last_err: Exception | None = None
        content = ""
        for attempt in range(3):
            try:
                content = self.chat_vision(prompt, image_paths, system, model=model)
                parsed = self._parse_json(content)
                logger.debug(
                    "chat_vision_json OK attempt %d/3 keys=%s",
                    attempt + 1,
                    list(parsed.keys()),
                )
                redis_cache.cache_set_sync(cache_key, parsed, self.settings.redis_cache_ttl_seconds, image_paths)
                return parsed
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                logger.warning(
                    "chat_vision_json attempt %d/3 failed: %s | raw response (first 500 chars): %r",
                    attempt + 1,
                    e,
                    content[:500],
                )
        raise ValueError(f"Model returned invalid JSON after 3 attempts: {last_err}")

    def chat_text_json(self, prompt: str, system: str = "") -> dict:
        content = self.chat_text(prompt, system)
        return self._parse_json(content)

    def embed(self, text: str) -> list[float]:
        cache_key = redis_cache.build_embed_cache_key(self.settings.ollama_embed_model, text)
        cached = redis_cache.cache_get_embed_sync(cache_key)
        if cached is not None:
            return cached
        response = self.embed_client.embed(model=self.settings.ollama_embed_model, input=text)
        embedding = response["embeddings"][0]
        redis_cache.cache_set_embed_sync(
            cache_key, embedding, self.settings.redis_cache_ttl_seconds
        )
        return embedding
