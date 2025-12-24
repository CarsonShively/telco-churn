import logging
from time import perf_counter

import duckdb

from telco_churn.io.hf import download_dataset_hf
from telco_churn.db.executor import SQLExecutor
from telco_churn.data_layers.bronze.ingest import build_bronze

from telco_churn.redis.connect import redis_config, connect_redis
from telco_churn.redis.writer import write_to_redis

from telco_churn.logging_utils import setup_logging

from telco_churn.config import (
    REPO_ID,
    BRONZE_ONLINE_PARQUET,
    DUCKDB_PATH,
    FEATURES_TABLE,
    ENTITY_COL,
)

SILVER_SQL_PKG = "telco_churn.data_layers.silver"
GOLD_SQL_PKG = "telco_churn.data_layers.gold"

BASE_SQL_FILE = "base.sql"
FEATURES_SQL_FILE = "features.sql"

REDIS_WRITE_BATCH_SIZE = 1000

log = logging.getLogger(__name__)


def main() -> None:
    t0 = perf_counter()
    log.info("pipeline start")

    with duckdb.connect(DUCKDB_PATH) as con:
        ex = SQLExecutor(con)

        log.info("Downloading bronze data")
        local_path = download_dataset_hf(
            repo_id=REPO_ID,
            filename=BRONZE_ONLINE_PARQUET,
        )

        log.info("Building bronze tables")
        build_bronze(con, local_path)

        log.info("Running SQL stage: base")
        ex.execute_script(ex.load_sql(SILVER_SQL_PKG, BASE_SQL_FILE))

        log.info("Running SQL stage: features")
        ex.execute_script(ex.load_sql(GOLD_SQL_PKG, FEATURES_SQL_FILE))

        log.info("Connecting to Redis")
        redis_cfg = redis_config()
        r = connect_redis(redis_cfg)

        log.info(
            "Writing features to Redis (table=%s, batch_size=%s)",
            FEATURES_TABLE,
            REDIS_WRITE_BATCH_SIZE,
        )

        written, run_prefix = write_to_redis(
            con,
            r,
            cfg=redis_cfg,
            table=FEATURES_TABLE,
            entity_col=ENTITY_COL,
            batch_size=REDIS_WRITE_BATCH_SIZE,
        )

        log.info(
            "Redis write complete: entities=%s, run_prefix=%s",
            written,
            run_prefix,
        )

    log.info("pipeline completed in %.2fs", perf_counter() - t0)

if __name__ == "__main__":
    setup_logging("INFO")
    try:
        main()
    except Exception:
        log.exception("pipeline failed")
        raise
