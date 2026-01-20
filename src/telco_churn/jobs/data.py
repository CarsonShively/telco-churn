"""Build gold.train.parquet from bronze offline data using DuckDB SQL, optionally uploading to HF."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter

import duckdb

from telco_churn.io.hf import download_dataset_hf, upload_dataset_hf
from telco_churn.db.executor import SQLExecutor
from telco_churn.data_layers.bronze.ingest import build_bronze
from telco_churn.logging_utils import setup_logging
from telco_churn.config import (
    REPO_ID,
    BRONZE_OFFLINE_PARQUET,
    GOLD_TRAIN_PARQUET,
    DUCKDB_PATH,
)

log = logging.getLogger(__name__)

SILVER_SQL_PKG = "telco_churn.data_layers.silver"
GOLD_SQL_PKG = "telco_churn.data_layers.gold"

BASE_SQL_FILE = "base.sql"
LABEL_SQL_FILE = "label.sql"
FEATURES_SQL_FILE = "features.sql"
TRAIN_SQL_FILE = "train.sql"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--upload", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main(*, upload: bool) -> None:
    """Run the training dataset build pipeline and optionally upload the output parquet to HF."""
    t0 = perf_counter()
    log.info("build_train start (upload=%s, duckdb_path=%s)", upload, DUCKDB_PATH)

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / GOLD_TRAIN_PARQUET
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Output parquet: %s", out_path)

    with duckdb.connect(DUCKDB_PATH) as con:
        ex = SQLExecutor(con)

        log.info("Downloading bronze: repo=%s file=%s", REPO_ID, BRONZE_OFFLINE_PARQUET)
        local_bronze = download_dataset_hf(repo_id=REPO_ID, filename=BRONZE_OFFLINE_PARQUET)
        log.info("Bronze local path: %s", local_bronze)

        log.info("Building bronze")
        build_bronze(con, local_bronze)

        log.info("Running SQL stage: base")
        ex.execute_script(ex.load_sql(SILVER_SQL_PKG, BASE_SQL_FILE))

        log.info("Running SQL stage: label")
        ex.execute_script(ex.load_sql(SILVER_SQL_PKG, LABEL_SQL_FILE))

        log.info("Running SQL stage: features")
        ex.execute_script(ex.load_sql(GOLD_SQL_PKG, FEATURES_SQL_FILE))

        log.info("Running SQL stage: train")
        ex.execute_script(ex.load_sql(GOLD_SQL_PKG, TRAIN_SQL_FILE))

        nrows = con.execute("SELECT COUNT(*) FROM gold.train").fetchone()[0]
        log.info("gold.train rows: %s", nrows)

        if nrows == 0:
            raise RuntimeError("gold.train is empty (0 rows)")

        log.info("Writing parquet")
        ex.write_parquet("SELECT * FROM gold.train", str(out_path))

    if out_path.exists():
        log.info("Wrote parquet size_bytes=%s", out_path.stat().st_size)

    if upload:
        log.info("Uploading to HF: repo=%s dest=%s", REPO_ID, GOLD_TRAIN_PARQUET)
        upload_dataset_hf(local_path=str(out_path), repo_id=REPO_ID, hf_path=GOLD_TRAIN_PARQUET)
        log.info("Upload complete")

    log.info("build_train done in %.2fs", perf_counter() - t0)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)

    try:
        main(upload=args.upload)
    except Exception:
        log.exception("build_train failed")
        raise
