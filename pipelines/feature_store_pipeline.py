import os
from dataclasses import dataclass
from pathlib import Path

import duckdb

from telco_churn.io.hf import download_from_hf
from telco_churn.executor.executor_sql import SQLExecutor
from telco_churn.data_layers.bronze.bronze import build_bronze_raw_from_parquet

from telco_churn.redis.redis import RedisConfig, connect_redis, write_to_redis

BASE_SQL_PKG = "telco_churn.data_layers.silver"
BASE_SQL_FILE = "base.sql"

FEATURES_SQL_PKG = "telco_churn.data_layers.gold"
FEATURES_SQL_FILE = "features.sql"


@dataclass(frozen=True)
class PipelineConfig:
    repo_id: str = os.getenv("TELCO_REPO_ID", "Carson-Shively/telco-churn")
    online_data: str = os.getenv("TELCO_BRONZE_HF_PATH", "data/bronze/online.parquet")
    duckdb_path: str = os.getenv("TELCO_DUCKDB_PATH", ":memory:")

    redis_host: str = os.getenv("TELCO_REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("TELCO_REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("TELCO_REDIS_DB", "0"))

    redis_base_prefix: str = os.getenv("TELCO_REDIS_PREFIX", "telco:features:")
    redis_current_pointer_key: str = os.getenv("TELCO_REDIS_CURRENT_KEY", "telco:features:CURRENT")
    redis_run_meta_prefix: str = os.getenv("TELCO_REDIS_RUN_META_PREFIX", "telco:features:RUN_META:")

    redis_ttl_seconds: int = int(os.getenv("TELCO_REDIS_TTL_SECONDS", "0"))

    features_table: str = os.getenv("TELCO_FEATURES_TABLE", "gold.features")
    entity_col: str = os.getenv("TELCO_ENTITY_COL", "customer_id")


def main(cfg: PipelineConfig = PipelineConfig()) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    with duckdb.connect(cfg.duckdb_path) as con:
        ex = SQLExecutor(con)

        local_path = download_from_hf(repo_id=cfg.repo_id, filename=cfg.online_data)
        build_bronze_raw_from_parquet(con, local_path)

        ex.execute_script(ex.load_sql(BASE_SQL_PKG, BASE_SQL_FILE))
        ex.execute_script(ex.load_sql(FEATURES_SQL_PKG, FEATURES_SQL_FILE))

        ttl: int | None = None if cfg.redis_ttl_seconds <= 0 else cfg.redis_ttl_seconds

        r = connect_redis(
            RedisConfig(
                host=cfg.redis_host,
                port=cfg.redis_port,
                db=cfg.redis_db,
                base_prefix=cfg.redis_base_prefix,
                current_pointer_key=cfg.redis_current_pointer_key,
                run_meta_prefix=cfg.redis_run_meta_prefix,
                ttl_seconds=ttl,
            )
        )

        written, run_prefix = write_to_redis(
            con,
            r,
            table=cfg.features_table,
            entity_col=cfg.entity_col,
            base_prefix=cfg.redis_base_prefix,
            current_pointer_key=cfg.redis_current_pointer_key,
            run_meta_prefix=cfg.redis_run_meta_prefix,
            ttl_seconds=ttl,
            batch_size=1000,
        )

        print(f"Wrote {written} entities to Redis under prefix {run_prefix}")


if __name__ == "__main__":
    main()