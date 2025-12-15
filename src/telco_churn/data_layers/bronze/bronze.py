import duckdb

def build_bronze_raw_from_parquet(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
    *,
    table_name: str = "bronze.raw",
) -> str:
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
    con.execute(
        f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet(?)",
        [parquet_path],
    )
    return table_name