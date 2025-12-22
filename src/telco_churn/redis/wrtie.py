from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import duckdb
import redis

from telco_churn.redis.config import RedisConfig, make_run_prefix, make_entity_key


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def atomic_write_json(path: Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    atomic_write_text(path, text)


def write_to_redis(
    con: duckdb.DuckDBPyConnection,
    r: redis.Redis,
    *,
    cfg: RedisConfig,
    table: str,
    entity_col: str = "customer_id",
    batch_size: int = 1000,
) -> tuple[int, str]:
    started_at = datetime.now(timezone.utc).isoformat()

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_prefix = make_run_prefix(cfg.base_prefix, run_ts)
    run_meta_key = f"{cfg.run_meta_prefix}{run_ts}"

    ttl_seconds = cfg.ttl_seconds

    index_key = f"{run_prefix}customers:index"

    r.hset(
        run_meta_key,
        mapping={
            "status": "WRITING",
            "run_ts": run_ts,
            "run_prefix": run_prefix,
            "source_table": table,
            "entity_col": entity_col,
            "started_at": started_at,
            "index_key": index_key,
        },
    )

    written = 0
    try:
        cur = con.execute(f"SELECT * FROM {table}")
        cols = [d[0] for d in cur.description]

        if entity_col not in cols:
            raise ValueError(f"{table} must include entity column '{entity_col}'")

        entity_idx = cols.index(entity_col)

        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break

            pipe = r.pipeline(transaction=False)

            for row in rows:
                entity = row[entity_idx]
                if entity is None:
                    continue

                entity_id = str(entity)
                key = make_entity_key(run_prefix, entity_id)
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

                    pipe.sadd(index_key, entity_id)

                    if ttl_seconds is not None and ttl_seconds > 0:
                        pipe.expire(key, ttl_seconds)
                    written += 1

            pipe.execute()

        if ttl_seconds is not None and ttl_seconds > 0:
            r.expire(index_key, ttl_seconds)

        r.set(cfg.current_pointer_key, run_prefix)

        published_at = datetime.now(timezone.utc).isoformat()
        r.hset(
            run_meta_key,
            mapping={
                "status": "PUBLISHED",
                "entities_written": str(written),
                "published_at": published_at,
                "current_pointer_key": cfg.current_pointer_key,
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
                "entities_written": str(written),
            },
        )
        if ttl_seconds is not None and ttl_seconds > 0:
            r.expire(run_meta_key, ttl_seconds)
        raise
