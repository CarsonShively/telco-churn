import duckdb
from pathlib import Path
import dagster as dg
from telco_churn.paths import REPO_ROOT
from telco_churn.db.executor import SQLExecutor

@dg.asset(
    name="upload_train_table",
    required_resource_keys={"db", "hf_data"},
    config_schema={"upload": dg.Field(bool, default_value=False)},
)
def upload_train_table(context: dg.AssetExecutionContext, train_table: str):
    db = context.resources.db
    hf_data = context.resources.hf_data
    out_path = Path(REPO_ROOT / "data/gold/train.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)
        
        ex.write_parquet("SELECT * FROM gold.join_train", str(out_path))
        
        if context.op_config["upload"]:
            hf_data.upload_data(local_path=str(out_path), hf_path="data/gold/train.parquet")

        return str(out_path)