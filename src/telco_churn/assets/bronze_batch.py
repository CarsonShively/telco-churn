import dagster as dg
import duckdb
from telco_churn.data_layers.bronze.ingest import build_bronze


@dg.asset(name="bronze_batch_table", required_resource_keys={"db"})
def bronze_batch_table(context: dg.AssetExecutionContext, ingest_batch_data: str) -> str:
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        return build_bronze(con, ingest_batch_data, "bronze.batch")