import hashlib
import json
from typing import Any

import redis
import redis.asyncio as aioredis

from app.config.settings import Settings

_redis_async: aioredis.Redis | None = None
_redis_sync: redis.Redis | None = None


async def init_redis(settings: Settings) -> None:
    global _redis_async, _redis_sync
    if not settings.redis_url:
        return
    _redis_async = aioredis.from_url(settings.redis_url, decode_responses=True)
    _redis_sync = redis.from_url(settings.redis_url, decode_responses=True)
    await _redis_async.ping()
    _redis_sync.ping()


async def close_redis() -> None:
    global _redis_async, _redis_sync
    if _redis_async is not None:
        await _redis_async.aclose()
    if _redis_sync is not None:
        _redis_sync.close()
    _redis_async = None
    _redis_sync = None


def is_redis_available() -> bool:
    return _redis_sync is not None


def hash_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_vision_cache_key(prompt: str, image_paths: list) -> str:
    parts = [hash_text(prompt)]
    for p in image_paths:
        parts.append(hash_file(p))
    return hashlib.sha256(":".join(parts).encode()).hexdigest()


def cache_get_sync(key: str) -> dict | None:
    if _redis_sync is None:
        return None
    raw = _redis_sync.get(f"vision:{key}")
    if not raw:
        return None
    return json.loads(raw)


def cache_set_sync(key: str, value: Any, ttl_seconds: int) -> None:
    if _redis_sync is None:
        return
    _redis_sync.set(f"vision:{key}", json.dumps(value), ex=ttl_seconds)


def build_embed_cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}:{hash_text(text)}".encode()).hexdigest()


def cache_get_embed_sync(key: str) -> list[float] | None:
    if _redis_sync is None:
        return None
    raw = _redis_sync.get(f"embed:{key}")
    if not raw:
        return None
    return json.loads(raw)


def cache_set_embed_sync(key: str, value: list[float], ttl_seconds: int) -> None:
    if _redis_sync is None:
        return
    _redis_sync.set(f"embed:{key}", json.dumps(value), ex=ttl_seconds)


async def ping_redis() -> bool:
    if _redis_async is None:
        return False
    try:
        return (await _redis_async.ping()) is True
    except Exception:
        return False
