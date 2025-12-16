from __future__ import annotations

from typing import Iterable, Optional
import duckdb
import pandas as pd


def parquet_to_df(
    parquet_path: str,
    *,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    owns_con = con is None
    if con is None:
        con = duckdb.connect()

    try:
        if columns:
            cols_sql = ", ".join([f'"{c}"' for c in columns])
            query = f"SELECT {cols_sql} FROM read_parquet('{parquet_path}')"
        else:
            query = f"SELECT * FROM read_parquet('{parquet_path}')"

        return con.execute(query).fetchdf()
    finally:
        if owns_con:
            con.close()
