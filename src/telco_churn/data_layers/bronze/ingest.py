"""
Data parquet to DB table.
"""

import duckdb
from pathlib import Path

def build_bronze(con: duckdb.DuckDBPyConnection, parquet_path: str, bronze_table: str) -> str:
    if not Path(parquet_path).exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    con.execute("CREATE SCHEMA IF NOT EXISTS bronze;")
    con.execute(
        "CREATE OR REPLACE TABLE {bronze_table} AS SELECT * FROM read_parquet(?)".format(
            bronze_table=bronze_table
        ),
        [parquet_path],
    )


    return bronze_table
