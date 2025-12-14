import os
from dataclasses import dataclass
from pathlib import Path
import duckdb
from telco_churn.io.hf import download_from_hf
from telco_churn.executor.executor_sql import SQLExecutor

BRONZE_SQL_PKG = "telco_churn.data_layers.bronze"
BRONZE_SQL_FILE = "bronze.sql"

BASE_SQL_PKG = "telco_churn.data_layers.silver"
BASE_SQL_FILE = "base.sql"

LABEL_SQL_PKG = "telco_churn.data_layers.silver"
LABEL_SQL_FILE = "label.sql"

FEATURES_SQL_PKG = "telco_churn.data_layers.gold"
FEATURES_SQL_FILE = "features.sql"

TRAIN_SQL_PKG = "telco_churn.data_layers.gold"
TRAIN_SQL_FILE = "train.sql"

LOCAL_TRAIN_PATH = "data/gold/training.parquet"

@dataclass(frozen=True)
class PipelineConfig:
    repo_id: str = os.getenv("TELCO_REPO_ID", "Carson-Shively/telco-churn")
    bronze_hf_path: str = os.getenv("TELCO_BRONZE_HF_PATH", "data/bronze/offline.parquet")
    duckdb_path: str = os.getenv("TELCO_DUCKDB_PATH", ":memory:")

def main(cfg: PipelineConfig = PipelineConfig()) -> None:
    con = duckdb.connect(cfg.duckdb_path)
    ex = SQLExecutor(con)

    local_path = download_from_hf(repo_id=cfg.repo_id, filename=cfg.bronze_hf_path)

    bronze = ex.load_sql(BRONZE_SQL_PKG, BRONZE_SQL_FILE)
    ex.execute_script(bronze, (local_path,))

    base = ex.load_sql(BASE_SQL_PKG, BASE_SQL_FILE)
    ex.execute_script(base)
    
    label = ex.load_sql(LABEL_SQL_PKG, LABEL_SQL_FILE)
    ex.execute_script(label)
    
    features = ex.load_sql(FEATURES_SQL_PKG, FEATURES_SQL_FILE)
    ex.execute_script(features)
    
    train = ex.load_sql(TRAIN_SQL_PKG, TRAIN_SQL_FILE)
    ex.execute_script(train)
    
    out_path = Path(LOCAL_TRAIN_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ex.write_parquet("SELECT * FROM gold.training", str(out_path))
    
    if cfg.publish:
        upload_parquet(
            local_path=str(out_path),
            repo_id=cfg.repo_id,
            hf_path=cfg.train_hf_path,
        )

if __name__ == "__main__":
    main()