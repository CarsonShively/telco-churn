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
    redis_prefix: str = os.getenv("TELCO_REDIS_PREFIX", "telco:features:")
    redis_ttl_seconds: int = int(os.getenv("TELCO_REDIS_TTL_SECONDS", "0")) 

    features_table: str = os.getenv("TELCO_FEATURES_TABLE", "gold_features")  
    entity_col: str = os.getenv("TELCO_ENTITY_COL", "customer_id")


def main(cfg: PipelineConfig = PipelineConfig()) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    with duckdb.connect(cfg.duckdb_path) as con:
        ex = SQLExecutor(con)

        local_path = download_from_hf(repo_id=cfg.repo_id, filename=cfg.online_data)
        build_bronze_raw_from_parquet(con, local_path)

        ex.execute_script(ex.load_sql(BASE_SQL_PKG, BASE_SQL_FILE))
        ex.execute_script(ex.load_sql(FEATURES_SQL_PKG, FEATURES_SQL_FILE))

        r = connect_redis(
            RedisConfig(
                host=cfg.redis_host,
                port=cfg.redis_port,
                db=cfg.redis_db,
                prefix=cfg.redis_prefix,
                ttl_seconds=cfg.redis_ttl_seconds,
            )
        )

        write_to_redis(
            con,
            r,
            table=cfg.features_table,
            entity_col=cfg.entity_col,
            key_prefix=cfg.redis_prefix,
            ttl_seconds=cfg.redis_ttl_seconds,  
            batch_size=1000,
        )

if __name__ == "__main__":
    main()
