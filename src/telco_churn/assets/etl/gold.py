import duckdb
import dagster as dg
from telco_churn.db.executor import SQLExecutor

GOLD_SQL_PKG = "telco_churn.data_layers.gold"
FEATURES_SQL_FILE = "features.sql"

@dg.asset(name="gold_data_table", required_resource_keys={"db"})
def gold_data_table(context: dg.AssetExecutionContext, silver_data_table: str) -> str:
    """Engineered features data table."""
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        ex = SQLExecutor(con)

        template = ex.load_sql(GOLD_SQL_PKG, FEATURES_SQL_FILE)
        sql = template.format(
            features_table="gold.train_features",
            base_table=silver_data_table,
        )
        ex.execute_script(sql)

        table_name = "gold.train_features"

        rows = con.execute(f"select count(*) from {table_name}").fetchone()[0]
        cols = con.execute(f"describe {table_name}").fetchall()
        preview_df = con.execute(f"select * from {table_name} limit 5").df()

        context.add_output_metadata({
            "db_path": dg.MetadataValue.path(str(db.db_path())),
            "table": table_name,
            "source_table": silver_data_table,
            "rows": rows,
            "columns": len(cols),
            "schema": dg.MetadataValue.md(
                "\n".join([f"- `{name}`: {dtype}" for (name, dtype, *_rest) in cols])
            ),
            "preview": dg.MetadataValue.md(preview_df.to_markdown(index=False)),
        })

        return table_name
