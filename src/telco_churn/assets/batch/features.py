import duckdb
import dagster as dg
import pandas as pd

@dg.asset(name="batch_features_df", required_resource_keys={"db"})
def batch_features_df(context: dg.AssetExecutionContext, gold_batch_table: str) -> pd.DataFrame:
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        X = con.execute("SELECT * FROM gold.batch_features").df()

    return X