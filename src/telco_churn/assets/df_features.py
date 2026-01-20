import duckdb
import dagster as dg
import pandas as pd

@dg.asset(name="df_features", required_resource_keys={"db"})
def df_features(context: dg.AssetExecutionContext, batch_features_table: str) -> pd.DataFrame:
    db = context.resources.db
    with duckdb.connect(str(db.db_path())) as con:
        X = con.execute("SELECT * FROM gold.batch_features").df()

    return X