from __future__ import annotations

import os
import redis
from dataclasses import dataclass

@dataclass(frozen=True)
class RedisConfig:
    host: str
    port: int = 6379
    db: int = 0
    base_prefix: str = "telco:features:"
    current_pointer_key: str = "telco:features:CURRENT"
    run_meta_prefix: str = "telco:features:RUN_META:"
    ttl_seconds: int | None = None

    @staticmethod
    def from_env() -> "RedisConfig":
        ttl_raw = int(os.getenv("TELCO_REDIS_TTL_SECONDS", "0"))
        ttl: int | None = None if ttl_raw <= 0 else ttl_raw

        return RedisConfig(
            host=os.getenv("TELCO_REDIS_HOST", "localhost"),
            port=int(os.getenv("TELCO_REDIS_PORT", "6379")),
            db=int(os.getenv("TELCO_REDIS_DB", "0")),
            base_prefix=os.getenv("TELCO_REDIS_PREFIX", "telco:features:"),
            current_pointer_key=os.getenv("TELCO_REDIS_CURRENT_KEY", "telco:features:CURRENT"),
            run_meta_prefix=os.getenv("TELCO_REDIS_RUN_META_PREFIX", "telco:features:RUN_META:"),
            ttl_seconds=ttl,
        )



def connect_redis(cfg: RedisConfig) -> redis.Redis:
    r = redis.Redis(
        host=cfg.host,
        port=cfg.port,
        db=cfg.db,
        decode_responses=True,
    )
    r.ping()
    return r


def make_run_prefix(base_prefix: str, run_ts: str) -> str:
    return f"{base_prefix}v{run_ts}:"


def make_entity_key(run_prefix: str, entity: str) -> str:
    return f"{run_prefix}{entity}"
