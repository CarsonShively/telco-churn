import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import json

from telco_churn.io.hf import download_dataset_hf, read_model_json, load_model_hf, upload_dataset_hf
from telco_churn.config import REPO_ID, REVISION, BRONZE_ONLINE_PARQUET, DUCKDB_PATH
from telco_churn.data_layers.bronze.ingest import build_bronze
from telco_churn.db.executor import SQLExecutor
from telco_churn.batch.summary import build_batch_summary_core
from telco_churn.batch.latest_batch import write_latest_pointer
from telco_churn.batch.scored import build_scored_df
from telco_churn.batch.action import build_actions_df


SILVER_SQL_PKG = "telco_churn.data_layers.silver"
GOLD_SQL_PKG = "telco_churn.data_layers.gold"

BASE_SQL_FILE = "base.sql"
LABEL_SQL_FILE = "label.sql"
FEATURES_SQL_FILE = "features.sql"
TRAIN_SQL_FILE = "train.sql"

def main():
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    
    REPO_ROOT = Path(__file__).resolve().parents[1]
    reports_root = REPO_ROOT / "reports"
    batch_root = reports_root / f"batch_{batch_id}"
    batch_root.mkdir(parents=True, exist_ok=True)

    scored_path  = batch_root / "scored.parquet"
    actions_path = batch_root / "actions.parquet"
    summary_path = batch_root / "summary.json"
    
    hf_batch_path = f"reports/batch_{batch_id}"
    
    champion_ptr = read_model_json(repo_id=REPO_ID, revision=REVISION, path_in_repo="champion.json")
    model_version = champion_ptr["path_in_repo"]

    artifact = load_model_hf(
        repo_id=REPO_ID,
        revision=REVISION,
        path_in_repo=f'{champion_ptr["path_in_repo"]}/model.joblib',
    )
    model = getattr(artifact, "model", artifact)
    
    meta = read_model_json(repo_id=REPO_ID, revision=REVISION, path_in_repo=f'{champion_ptr["path_in_repo"]}/metadata.json')
    threshold = meta.get("cfg", {}).get("threshold")
    feature_names = meta.get("feature_names")
    
    with duckdb.connect(DUCKDB_PATH) as con:
        ex = SQLExecutor(con)
        
        batch = download_dataset_hf(repo_id=REPO_ID, revision=REVISION, filename=BRONZE_ONLINE_PARQUET)

        build_bronze(con, batch)

        ex.execute_script(ex.load_sql(SILVER_SQL_PKG, BASE_SQL_FILE))

        ex.execute_script(ex.load_sql(GOLD_SQL_PKG, FEATURES_SQL_FILE))
        
        X = con.execute("SELECT * FROM gold.features").df()

        proba = model.predict_proba(X)[:, 1]
        
        scored = build_scored_df(
            X=X,
            proba=proba,
            batch_id=batch_id,
            threshold=threshold,
        )

        scored.to_parquet(scored_path, index=False)
        
        actions = build_actions_df(
            scored=scored,
            X=X,
            model=model,
            names=feature_names
        )
        
        actions.to_parquet(actions_path, index=False)

        summary = build_batch_summary_core(
            batch_id=batch_id,
            model_version=model_version,
            threshold=threshold,
            scored=scored,
            actions=actions,
            top_k=3,
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        for p in [scored_path, actions_path, summary_path]:
            if not p.exists():
                raise RuntimeError(f"Missing output file: {p}")
        
        upload_dataset_hf(
            local_path=str(scored_path),
            repo_id=REPO_ID,
            hf_path=f"{hf_batch_path}/scored.parquet",
            revision=REVISION,
        )

        upload_dataset_hf(
            local_path=str(actions_path),
            repo_id=REPO_ID,
            hf_path=f"{hf_batch_path}/actions.parquet",
            revision=REVISION,
        )

        upload_dataset_hf(
            local_path=str(summary_path),
            repo_id=REPO_ID,
            hf_path=f"{hf_batch_path}/summary.json",
            revision=REVISION,
        )
        
        write_latest_pointer(reports_root=reports_root, batch_id=batch_id)
        
        upload_dataset_hf(
            local_path=str(reports_root / "latest.json"),
            repo_id=REPO_ID,
            hf_path="reports/latest.json",
            revision=REVISION,
        )
        
if __name__ == "__main__":
    main()