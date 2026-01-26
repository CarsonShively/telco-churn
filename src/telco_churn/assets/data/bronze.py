import dagster as dg
import duckdb
from telco_churn.data_layers.bronze.ingest import build_bronze

@dg.asset(name="bronze_data_table", required_resource_keys={"db"})
def bronze_data_table(context: dg.AssetExecutionContext, raw_data: str) -> str:
    """Raw data to duckdb table."""
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        table_name = build_bronze(con, raw_data, "bronze.train")

        rows = con.execute(f"select count(*) from {table_name}").fetchone()[0]
        cols = con.execute(f"describe {table_name}").fetchall()
        preview_df = con.execute(f"select * from {table_name} limit 5").df()

        context.add_output_metadata({
            "db_path": dg.MetadataValue.path(str(db.db_path())),
            "table": table_name,
            "source_path": raw_data,
            "rows": rows,
            "columns": len(cols),
            "schema": dg.MetadataValue.md(
                "\n".join([f"- `{name}`: {dtype}" for (name, dtype, *_rest) in cols])
            ),
            "preview": dg.MetadataValue.md(preview_df.to_markdown(index=False)),
        })

        return table_name
