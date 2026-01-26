import duckdb
from pathlib import Path
import os
import dagster as dg
from telco_churn.paths import REPO_ROOT
from telco_churn.db.executor import SQLExecutor

@dg.asset(
    name="upload_train_table",
    required_resource_keys={"db", "hf_data"},
    config_schema={"upload": dg.Field(bool, default_value=False)},
)
def upload_train_table(context: dg.AssetExecutionContext, train_table: str) -> str:
    """Upload model training ready data to hugging face data archive."""
    db = context.resources.db
    hf_data = context.resources.hf_data

    out_path = Path(REPO_ROOT / "data/gold/train.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)
        ex.write_parquet("SELECT * FROM gold.join_train", str(out_path))

    uploaded = False
    hf_path = None

    if context.op_config["upload"]:
        hf_path = "data/gold/train.parquet"
        hf_data.upload_data(local_path=str(out_path), hf_path=hf_path)
        uploaded = True

    context.add_output_metadata({
        "table": train_table,
        "local_path": dg.MetadataValue.path(str(out_path)),
        "bytes": os.path.getsize(out_path),
        "uploaded": uploaded,
        **({"hf_path": hf_path} if uploaded else {}),
    })

    return str(out_path)
