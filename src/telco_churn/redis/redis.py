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
    base_prefix: str = "telco:features:" 
    current_pointer_key: str = "telco:features:CURRENT"
    run_meta_prefix: str = "telco:features:RUN_META:"
    ttl_seconds: int | None = None  


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


def write_to_redis(
    con: duckdb.DuckDBPyConnection,
    r: redis.Redis,
    *,
    table: str,
    entity_col: str = "customer_id",
    base_prefix: str = "telco:features:",
    current_pointer_key: str = "telco:features:CURRENT",
    run_meta_prefix: str = "telco:features:RUN_META:",
    ttl_seconds: int | None = None, 
    batch_size: int = 1000,
) -> tuple[int, str]:
    started_at = datetime.now(timezone.utc).isoformat()

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = make_run_prefix(base_prefix, run_ts)
    run_meta_key = f"{run_meta_prefix}{run_ts}"

    r.hset(
        run_meta_key,
        mapping={
            "status": "WRITING",
            "run_ts": run_ts,
            "run_prefix": run_prefix,
            "source_table": table,
            "entity_col": entity_col,
            "started_at": started_at,
        },
    )

    written = 0
    try:
        cur = con.execute(f"SELECT * FROM {table}")
        cols = [d[0] for d in cur.description]

        if entity_col not in cols:
            raise ValueError(f"{table} must include entity column '{entity_col}'")

        entity_idx = cols.index(entity_col)
        pipe = r.pipeline(transaction=False)

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                entity = row[entity_idx]
                if entity is None:
                    continue

                key = f"{run_prefix}{entity}"
                mapping: dict[str, str] = {}

                for i, col in enumerate(cols):
                    if col == entity_col:
                        continue
                    val: Any = row[i]
                    if val is None:
                        continue
                    mapping[col] = str(val)

                if mapping:
                    pipe.hset(key, mapping=mapping)
                    if ttl_seconds is not None and ttl_seconds > 0:
                        pipe.expire(key, ttl_seconds)
                    written += 1

            pipe.execute()

        r.set(current_pointer_key, run_prefix)

        published_at = datetime.now(timezone.utc).isoformat()
        r.hset(
            run_meta_key,
            mapping={
                "status": "PUBLISHED",
                "rows_written": str(written),
                "published_at": published_at,
                "current_pointer_key": current_pointer_key,
            },
        )

        if ttl_seconds is not None and ttl_seconds > 0:
            r.expire(run_meta_key, ttl_seconds)

        return written, run_prefix

    except Exception as e:
        r.hset(
            run_meta_key,
            mapping={
                "status": "FAILED",
                "error": repr(e),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "rows_written": str(written),
            },
        )
        if ttl_seconds is not None and ttl_seconds > 0:
            r.expire(run_meta_key, ttl_seconds)
        raise
