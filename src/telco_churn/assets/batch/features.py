import duckdb
import dagster as dg
import pandas as pd

@dg.asset(name="batch_features_df", required_resource_keys={"db"})
def batch_features_df(context: dg.AssetExecutionContext, gold_batch_table: str) -> pd.DataFrame:
    """Batch feature set dataframe."""
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        X = con.execute("SELECT * FROM gold.batch_features").df()

    context.add_output_metadata({
        "db_path": dg.MetadataValue.path(str(db.db_path())),
        "source_table": gold_batch_table,
        "rows": X.shape[0],
        "columns": X.shape[1],
        "preview": dg.MetadataValue.md(X.head(5).to_markdown(index=False)),
        "memory_bytes": int(X.memory_usage(deep=True).sum()),
    })

    return X
