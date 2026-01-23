import duckdb
import dagster as dg
from telco_churn.resources.duckdb import DuckDBResource
from telco_churn.db.executor import SQLExecutor

GOLD_SQL_PKG = "telco_churn.data_layers.gold"
TRAIN_SQL_FILE = "train.sql"

@dg.asset(name="train_table", required_resource_keys={"db"})
def train_table(context: dg.AssetExecutionContext, gold_data_table: str, labels_table: str) -> str:
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)

        ex.execute_script(ex.load_sql(GOLD_SQL_PKG, TRAIN_SQL_FILE))

        return "gold.join_train"