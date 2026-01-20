import duckdb
from pathlib import Path
import dagster as dg

from telco_churn.db.executor import SQLExecutor

@dg.asset(
    name="materialise_gold",
    required_resource_keys={"db", "hf_data"},
    config_schema={"upload": dg.Field(bool, default_value=False)},
)
def materiailise_gold(context: dg.AssetExecutionContext, train_join_table: str):
    db = context.resources.db
    hf_data = context.resources.hf_data
    repo_root = Path(__file__).resolve().parents[3]
    out_path = repo_root / "data/gold/train.parquet"
    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)
        
        ex.write_parquet("SELECT * FROM gold.join_train", str(out_path))
        
        if context.op_config["upload"]:
            hf_data.upload_data(local_path=str(out_path), hf_path="data/gold/train.parquet")

        return str(out_path)