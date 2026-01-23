import duckdb
import dagster as dg
from telco_churn.db.executor import SQLExecutor

SILVER_SQL_PKG = "telco_churn.data_layers.silver"
LABEL_SQL_FILE = "label.sql"

@dg.asset(name="labels_table", required_resource_keys={"db"})
def labels_table(context: dg.AssetExecutionContext, bronze_data_table: str) -> str:
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)
        ex.execute_script(ex.load_sql(SILVER_SQL_PKG, LABEL_SQL_FILE))
        return "silver.labels"