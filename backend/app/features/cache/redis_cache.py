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


def cache_set_sync(key: str, value: Any, ttl_seconds: int, image_paths: list | None = None) -> None:
    if _redis_sync is None:
        return
    _redis_sync.set(f"vision:{key}", json.dumps(value), ex=ttl_seconds)
    # Store reverse index: image_hash → cache_key so we can invalidate by image
    if image_paths:
        for p in image_paths:
            img_hash = hash_file(p)
            _redis_sync.sadd(f"img_keys:{img_hash}", f"vision:{key}")
            _redis_sync.expire(f"img_keys:{img_hash}", ttl_seconds)


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


def invalidate_image_cache(image_path) -> int:
    """
    Delete all vision cache entries that were built from a given image file.
    Called when a user submits a correction so the next upload of the same
    sketch goes through the full pipeline instead of returning a stale result.
    """
    if _redis_sync is None:
        return 0
    img_hash = hash_file(image_path)
    index_key = f"img_keys:{img_hash}"
    cache_keys = _redis_sync.smembers(index_key)
    if not cache_keys:
        return 0
    deleted = _redis_sync.delete(*cache_keys, index_key)
    return deleted


async def ping_redis() -> bool:
    if _redis_async is None:
        return False
    try:
        return (await _redis_async.ping()) is True
    except Exception:
        return False
