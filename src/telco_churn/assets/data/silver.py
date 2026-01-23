import duckdb
import dagster as dg
from telco_churn.db.executor import SQLExecutor

SILVER_SQL_PKG = "telco_churn.data_layers.silver"
BASE_SQL_FILE = "base.sql"

@dg.asset(name="silver_data_table", required_resource_keys={"db"})
def silver_data_table(context: dg.AssetExecutionContext, bronze_data_table: str) -> str:
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)

        template = ex.load_sql(SILVER_SQL_PKG, BASE_SQL_FILE)
        sql = template.format(
            base_table="silver.train_base",
            bronze_table=bronze_data_table,
        )
        ex.execute_script(sql)

        return "silver.train_base"