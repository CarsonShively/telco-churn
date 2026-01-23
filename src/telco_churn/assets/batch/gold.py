import duckdb
import dagster as dg
from telco_churn.db.executor import SQLExecutor

GOLD_SQL_PKG = "telco_churn.data_layers.gold"
FEATURES_SQL_FILE = "features.sql"

@dg.asset(name="gold_batch_table", required_resource_keys={"db"})
def gold_batch_table(context: dg.AssetExecutionContext, silver_batch_table: str) -> str:
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)

        template = ex.load_sql(GOLD_SQL_PKG, FEATURES_SQL_FILE)
        sql = template.format(
            features_table="gold.batch_features",
            base_table=silver_batch_table,
        )
        ex.execute_script(sql)

        return "gold.batch_features"