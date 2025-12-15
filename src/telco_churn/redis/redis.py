from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from datetime import datetime, timezone

import duckdb
import redis


@dataclass(frozen=True)
class RedisConfig:
    host: str
    port: int = 6379
    db: int = 0
    prefix: str = "telco:features:"
    ttl_seconds: int = 0  # 0 = no TTL


def connect_redis(cfg: RedisConfig) -> redis.Redis:
    r = redis.Redis(
        host=cfg.host,
        port=cfg.port,
        db=cfg.db,
        decode_responses=True,
    )
    r.ping()
    return r


def write_to_redis(
    con: duckdb.DuckDBPyConnection,
    r: redis.Redis,
    *,
    table: str,
    entity_col: str = "customer_id",
    key_prefix: str | None = None,
    ttl_seconds: int | None = None,
    batch_size: int = 1000,
    ts_field: str = "__ts",
) -> int:
    key_prefix = key_prefix if key_prefix is not None else "telco:features:"
    ttl_seconds = ttl_seconds if ttl_seconds is not None else 0

    # One timestamp for the whole materialization run (recommended)
    run_ts = datetime.now(timezone.utc).isoformat()

    cur = con.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cur.description]

    if entity_col not in cols:
        raise ValueError(f"{table} must include entity column '{entity_col}'")

    entity_idx = cols.index(entity_col)

    pipe = r.pipeline(transaction=False)
    written = 0

    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break

        for row in rows:
            entity = row[entity_idx]
            if entity is None:
                continue

            key = f"{key_prefix}{entity}"

            mapping: dict[str, str] = {ts_field: run_ts}  # <-- freshness metadata

            for i, col in enumerate(cols):
                if col == entity_col:
                    continue
                val: Any = row[i]
                if val is None:
                    continue
                mapping[col] = str(val)

            # If you want to skip entities with zero real features (besides __ts):
            if len(mapping) > 1:
                pipe.hset(key, mapping=mapping)
                if ttl_seconds > 0:
                    pipe.expire(key, ttl_seconds)
                written += 1

        pipe.execute()

    return written
