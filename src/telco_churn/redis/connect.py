"""Redis connection helpers for building a RedisConfig from app settings and creating a validated Redis client."""

from __future__ import annotations

import redis

from telco_churn.redis.infra import RedisConfig
from telco_churn.config import (
    REDIS_HOST, REDIS_PORT, REDIS_DB,
    REDIS_BASE_PREFIX, REDIS_CURRENT_POINTER_KEY, REDIS_RUN_META_PREFIX,
    REDIS_TTL_SECONDS,
)

def redis_config() -> RedisConfig:
    return RedisConfig(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        base_prefix=REDIS_BASE_PREFIX,
        current_pointer_key=REDIS_CURRENT_POINTER_KEY,
        run_meta_prefix=REDIS_RUN_META_PREFIX,
        ttl_seconds=None if REDIS_TTL_SECONDS in (0, None) else int(REDIS_TTL_SECONDS),
    )

def connect_redis(cfg: RedisConfig) -> redis.Redis:
    r = redis.Redis(host=cfg.host, port=cfg.port, db=cfg.db, decode_responses=True)
    r.ping()
    return r
