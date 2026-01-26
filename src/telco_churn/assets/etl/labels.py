import duckdb
import dagster as dg
from telco_churn.db.executor import SQLExecutor

SILVER_SQL_PKG = "telco_churn.data_layers.silver"
LABEL_SQL_FILE = "label.sql"

@dg.asset(name="labels_table", required_resource_keys={"db"})
def labels_table(context: dg.AssetExecutionContext, bronze_data_table: str) -> str:
    """Churn labels data table."""
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)
        ex.execute_script(ex.load_sql(SILVER_SQL_PKG, LABEL_SQL_FILE))

        table_name = "silver.labels"

        rows = con.execute(f"select count(*) from {table_name}").fetchone()[0]
        cols = con.execute(f"describe {table_name}").fetchall()
        preview_df = con.execute(f"select * from {table_name} limit 5").df()

        context.add_output_metadata({
            "db_path": dg.MetadataValue.path(str(db.db_path())),
            "table": table_name,
            "source_table": bronze_data_table,
            "rows": rows,
            "columns": len(cols),
            "schema": dg.MetadataValue.md(
                "\n".join([f"- `{name}`: {dtype}" for (name, dtype, *_rest) in cols])
            ),
            "preview": dg.MetadataValue.md(preview_df.to_markdown(index=False)),
        })

        return table_name
