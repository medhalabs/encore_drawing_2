import base64
import json
import re
from pathlib import Path

from ollama import Client

from app.config.settings import Settings
from app.features.cache import redis_cache


class OllamaService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = Client(
            host=settings.ollama_base_url,
            headers={"Authorization": f"Bearer {settings.ollama_api_key}"},
        )

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

    def chat_vision(self, prompt: str, image_paths: list[Path], system: str = "") -> str:
        images = [self._image_to_base64(p) for p in image_paths]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": images})
        response = self.client.chat(
            model=self.settings.ollama_vision_model,
            messages=messages,
            stream=False,
        )
        return response["message"]["content"]

    def chat_vision_json(self, prompt: str, image_paths: list[Path], system: str = "") -> dict:
        cache_key = redis_cache.build_vision_cache_key(prompt, image_paths)
        cached = redis_cache.cache_get_sync(cache_key)
        if cached is not None:
            return cached
        content = self.chat_vision(prompt, image_paths, system)
        parsed = self._parse_json(content)
        redis_cache.cache_set_sync(cache_key, parsed, self.settings.redis_cache_ttl_seconds)
        return parsed

    def chat_text_json(self, prompt: str, system: str = "") -> dict:
        content = self.chat_text(prompt, system)
        return self._parse_json(content)

    def embed(self, text: str) -> list[float]:
        cache_key = redis_cache.build_embed_cache_key(self.settings.ollama_embed_model, text)
        cached = redis_cache.cache_get_embed_sync(cache_key)
        if cached is not None:
            return cached
        response = self.client.embed(model=self.settings.ollama_embed_model, input=text)
        embedding = response["embeddings"][0]
        redis_cache.cache_set_embed_sync(
            cache_key, embedding, self.settings.redis_cache_ttl_seconds
        )
        return embedding
