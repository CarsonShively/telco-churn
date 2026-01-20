import duckdb
import dagster as dg
from telco_churn.db.executor import SQLExecutor

GOLD_SQL_PKG = "telco_churn.data_layers.gold"
FEATURES_SQL_FILE = "features.sql"

@dg.asset(name="batch_features_table", required_resource_keys={"db"})
def batch_features_table(context: dg.AssetExecutionContext, batch_base_table: str) -> str:
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)

        template = ex.load_sql(GOLD_SQL_PKG, FEATURES_SQL_FILE)
        sql = template.format(
            features_table="gold.batch_features",
            base_table=batch_base_table,
        )
        ex.execute_script(sql)

        return "gold.batch_features"