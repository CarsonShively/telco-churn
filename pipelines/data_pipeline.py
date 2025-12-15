import os
from dataclasses import dataclass
from pathlib import Path

import duckdb

from telco_churn.io.hf import download_from_hf, upload_parquet
from telco_churn.executor.executor_sql import SQLExecutor
from telco_churn.data_layers.bronze.bronze import build_bronze_raw_from_parquet

BASE_SQL_PKG = "telco_churn.data_layers.silver"
BASE_SQL_FILE = "base.sql"

LABEL_SQL_PKG = "telco_churn.data_layers.silver"
LABEL_SQL_FILE = "label.sql"

FEATURES_SQL_PKG = "telco_churn.data_layers.gold"
FEATURES_SQL_FILE = "features.sql"

TRAIN_SQL_PKG = "telco_churn.data_layers.gold"
TRAIN_SQL_FILE = "train.sql"


@dataclass(frozen=True)
class PipelineConfig:
    repo_id: str = os.getenv("TELCO_REPO_ID", "Carson-Shively/telco-churn")
    bronze_hf_path: str = os.getenv("TELCO_BRONZE_HF_PATH", "data/bronze/offline.parquet")
    train_hf_path: str = os.getenv("TELCO_TRAIN_HF_PATH", "data/gold/training.parquet")
    duckdb_path: str = os.getenv("TELCO_DUCKDB_PATH", ":memory:")
    publish: bool = os.getenv("TELCO_PUBLISH", "0") == "1"


def main(cfg: PipelineConfig = PipelineConfig()) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "data" / "gold" / "training.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(cfg.duckdb_path) as con:
        ex = SQLExecutor(con)

        local_path = download_from_hf(repo_id=cfg.repo_id, filename=cfg.bronze_hf_path)
        build_bronze_raw_from_parquet(con, local_path)

        ex.execute_script(ex.load_sql(BASE_SQL_PKG, BASE_SQL_FILE))
        ex.execute_script(ex.load_sql(LABEL_SQL_PKG, LABEL_SQL_FILE))
        ex.execute_script(ex.load_sql(FEATURES_SQL_PKG, FEATURES_SQL_FILE))
        ex.execute_script(ex.load_sql(TRAIN_SQL_PKG, TRAIN_SQL_FILE))

        ex.write_parquet("SELECT * FROM gold.training", str(out_path))

    if cfg.publish:
        upload_parquet(local_path=str(out_path), repo_id=cfg.repo_id, hf_path=cfg.train_hf_path)


if __name__ == "__main__":
    main()
