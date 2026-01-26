import duckdb
import dagster as dg
from telco_churn.db.executor import SQLExecutor

GOLD_SQL_PKG = "telco_churn.data_layers.gold"
TRAIN_SQL_FILE = "train.sql"

@dg.asset(name="train_table", required_resource_keys={"db"})
def train_table(
    context: dg.AssetExecutionContext,
    gold_data_table: str,
    labels_table: str,
) -> str:
    """Train ready data table."""
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)
        ex.execute_script(ex.load_sql(GOLD_SQL_PKG, TRAIN_SQL_FILE))

        table_name = "gold.join_train"

        rows = con.execute(f"select count(*) from {table_name}").fetchone()[0]
        cols = con.execute(f"describe {table_name}").fetchall()
        preview_df = con.execute(f"select * from {table_name} limit 5").df()

        context.add_output_metadata({
            "db_path": dg.MetadataValue.path(str(db.db_path())),
            "table": table_name,
            "source_tables": [gold_data_table, labels_table],
            "rows": rows,
            "columns": len(cols),
            "schema": dg.MetadataValue.md(
                "\n".join([f"- `{name}`: {dtype}" for (name, dtype, *_rest) in cols])
            ),
            "preview": dg.MetadataValue.md(preview_df.to_markdown(index=False)),
        })

        return table_name
