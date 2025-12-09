"""
Chunk summary caching by content hash.

Caches chunk summaries to avoid re-processing identical chunks.
Uses Redis if available, otherwise falls back to in-memory cache.
"""
import hashlib
import json
import time
from typing import Optional, Dict
import os
import threading

# Use Redis if available, otherwise in-memory dict
USE_REDIS = False
cache_client = None
_memory_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()

try:
    import redis
    USE_REDIS = True
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    try:
        cache_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=0,
            decode_responses=True,
            socket_connect_timeout=2  # Quick timeout to avoid blocking if Redis is down
        )
        # Test connection
        cache_client.ping()
        print(f"✓ Chunk cache: Using Redis at {redis_host}:{redis_port}")
    except (redis.ConnectionError, redis.TimeoutError, Exception) as e:
        print(f"⚠️  Chunk cache: Redis unavailable ({e}), using in-memory cache")
        USE_REDIS = False
        cache_client = None
except ImportError:
    USE_REDIS = False
    print("ℹ️  Chunk cache: Redis not installed, using in-memory cache (pip install redis)")


def get_cache_key(chunk: str, prompt_template: str, model: str) -> str:
    """
    Generate cache key from content hash, prompt hash, and model.
    
    Args:
        chunk: The content chunk
        prompt_template: The prompt template used
        model: The model name
    
    Returns:
        Cache key string
    """
    content_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
    prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()[:16]
    return f"chunk_summary:{content_hash}:{prompt_hash}:{model}"


def get_cached_summary(chunk: str, prompt_template: str, model: str) -> Optional[dict]:
    """
    Get cached chunk summary if available.
    
    Args:
        chunk: The content chunk
        prompt_template: The prompt template used
        model: The model name
    
    Returns:
        Cached summary dict (with 'summary', 'key_points', 'entities') or None
    """
    key = get_cache_key(chunk, prompt_template, model)
    
    if USE_REDIS and cache_client:
        try:
            cached = cache_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            print(f"⚠️  Cache read error: {e}")
            return None
    else:
        # In-memory cache
        with _cache_lock:
            entry = _memory_cache.get(key)
            if entry:
                # Check TTL
                if time.time() - entry["timestamp"] < entry["ttl"]:
                    return entry["data"]
                else:
                    # Expired, remove it
                    del _memory_cache[key]
    
    return None


def cache_summary(chunk: str, prompt_template: str, model: str, summary: dict, ttl_days: int = 30):
    """
    Cache chunk summary.
    
    Args:
        chunk: The content chunk
        prompt_template: The prompt template used
        model: The model name
        summary: Summary dict to cache (should have 'summary', 'key_points', 'entities')
        ttl_days: Time to live in days (default: 30)
    """
    key = get_cache_key(chunk, prompt_template, model)
    ttl_seconds = ttl_days * 86400
    
    if USE_REDIS and cache_client:
        try:
            cache_client.setex(key, ttl_seconds, json.dumps(summary))
        except Exception as e:
            print(f"⚠️  Cache write error: {e}")
    else:
        # In-memory cache
        with _cache_lock:
            _memory_cache[key] = {
                "data": summary,
                "timestamp": time.time(),
                "ttl": ttl_seconds
            }
            
            # Cleanup expired entries periodically (every 1000 writes)
            if len(_memory_cache) > 1000 and len(_memory_cache) % 1000 == 0:
                now = time.time()
                expired_keys = [
                    k for k, v in _memory_cache.items()
                    if now - v["timestamp"] >= v["ttl"]
                ]
                for k in expired_keys:
                    del _memory_cache[k]


def invalidate_cache(prompt_template: str = None, model: str = None):
    """
    Invalidate cache entries matching prompt or model.
    
    Args:
        prompt_template: If provided, invalidate entries with this prompt
        model: If provided, invalidate entries with this model
    """
    if USE_REDIS and cache_client:
        try:
            # Use pattern matching to delete keys
            pattern = "chunk_summary:*"
            if model:
                pattern = f"chunk_summary:*:*:{model}"
            elif prompt_template:
                prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()[:16]
                pattern = f"chunk_summary:*:{prompt_hash}:*"
            
            deleted = 0
            for key in cache_client.scan_iter(match=pattern):
                cache_client.delete(key)
                deleted += 1
            print(f"Invalidated {deleted} cache entries")
        except Exception as e:
            print(f"⚠️  Cache invalidation error: {e}")
    else:
        # In-memory cache
        with _cache_lock:
            if model:
                keys_to_delete = [
                    k for k in _memory_cache.keys()
                    if k.endswith(f":{model}")
                ]
            elif prompt_template:
                prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()[:16]
                keys_to_delete = [
                    k for k in _memory_cache.keys()
                    if f":{prompt_hash}:" in k
                ]
            else:
                keys_to_delete = list(_memory_cache.keys())
            
            for k in keys_to_delete:
                del _memory_cache[k]
            print(f"Invalidated {len(keys_to_delete)} cache entries")


def get_cache_stats() -> dict:
    """
    Get cache statistics.
    
    Returns:
        dict with cache stats (size, hits, misses, etc.)
    """
    if USE_REDIS and cache_client:
        try:
            # Count keys matching pattern
            pattern = "chunk_summary:*"
            count = sum(1 for _ in cache_client.scan_iter(match=pattern))
            return {
                "backend": "redis",
                "size": count,
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", 6379))
            }
        except Exception as e:
            return {"backend": "redis", "error": str(e)}
    else:
        with _cache_lock:
            # Clean expired entries
            now = time.time()
            expired = [
                k for k, v in _memory_cache.items()
                if now - v["timestamp"] >= v["ttl"]
            ]
            for k in expired:
                del _memory_cache[k]
            
            return {
                "backend": "memory",
                "size": len(_memory_cache),
                "ttl_days": 30
            }

