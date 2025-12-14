import os
from dataclasses import dataclass

import duckdb

from telco_churn.data_layers.bronze.online import read_parquet_arrow
from telco_churn.io.hf import download_from_hf
from telco_churn.executor.executor_sql import SQLExecutor
from telco_churn.data_layers.silver.online import build_silver_offline_from_bronze
from telco_churn.parity.parity_test import assert_silver_parity

BRONZE_SQL_PKG = "telco_churn.data_layers.bronze"
BRONZE_SQL_FILE = "offline.sql"
SILVER_SQL_PKG = "telco_churn.data_layers.silver"
SILVER_SQL_FILE = "offline.sql"

@dataclass(frozen=True)
class PipelineConfig:
    repo_id: str = os.getenv("TELCO_REPO_ID", "Carson-Shively/telco-churn")
    bronze_hf_path: str = os.getenv("TELCO_BRONZE_HF_PATH", "data/bronze/offline.parquet")
    duckdb_path: str = os.getenv("TELCO_DUCKDB_PATH", ":memory:")


def main(cfg: PipelineConfig = PipelineConfig()) -> None:
    con = duckdb.connect(cfg.duckdb_path)
    ex = SQLExecutor(con)

    local_path = download_from_hf(repo_id=cfg.repo_id, filename=cfg.bronze_hf_path)

    bronze_sql = ex.load_sql(BRONZE_SQL_PKG, BRONZE_SQL_FILE)
    ex.execute_script(bronze_sql, [local_path])

    bronze_online = read_parquet_arrow(local_path)

    silver_sql = ex.load_sql(SILVER_SQL_PKG, SILVER_SQL_FILE)
    ex.execute_script(silver_sql)
    silver_offline = ex.fetcharrow("SELECT * FROM silver.offline")

    silver_online = build_silver_offline_from_bronze(bronze_online)

    assert_silver_parity(silver_offline, silver_online)
    print("✅ Pipeline ran + silver parity passed")

if __name__ == "__main__":
    main()
