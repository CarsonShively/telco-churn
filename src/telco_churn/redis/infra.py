"""Redis infrastructure primitives for the telco-churn feature store."""

from __future__ import annotations

from dataclasses import dataclass
import redis


@dataclass(frozen=True)
class RedisConfig:
    """Configuration schema for Redis feature store access."""
    host: str
    port: int = 6379
    db: int = 0
    base_prefix: str = "telco:features:"
    current_pointer_key: str = "telco:features:CURRENT"
    run_meta_prefix: str = "telco:features:RUN_META:"
    ttl_seconds: int | None = None


def connect_redis(cfg: RedisConfig) -> redis.Redis:
    """Create and validate a Redis client from the given configuration."""
    r = redis.Redis(
        host=cfg.host,
        port=cfg.port,
        db=cfg.db,
        decode_responses=True,
    )
    r.ping()
    return r


def make_run_prefix(base_prefix: str, run_ts: str) -> str:
    """Return a versioned Redis key prefix for a feature store run."""
    return f"{base_prefix}v{run_ts}:"


def make_entity_key(run_prefix: str, entity: str) -> str:
    """Return the Redis key for a single entity under a run prefix."""
    return f"{run_prefix}{entity}"
