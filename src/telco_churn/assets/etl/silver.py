import duckdb
import dagster as dg
from telco_churn.db.executor import SQLExecutor

SILVER_SQL_PKG = "telco_churn.data_layers.silver"
BASE_SQL_FILE = "base.sql"

@dg.asset(name="silver_data_table", required_resource_keys={"db"})
def silver_data_table(context: dg.AssetExecutionContext, bronze_data_table: str) -> str:
    """Normalised and cleaned features data table."""
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)

        template = ex.load_sql(SILVER_SQL_PKG, BASE_SQL_FILE)
        sql = template.format(
            base_table="silver.train_base",
            bronze_table=bronze_data_table,
        )
        ex.execute_script(sql)

        table_name = "silver.train_base"

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
