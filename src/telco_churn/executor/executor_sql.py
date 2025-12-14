from dataclasses import dataclass
from importlib.resources import files
import duckdb
import pandas as pd
import pyarrow as pa
from typing import Any


@dataclass
class SQLExecutor:
    con: duckdb.DuckDBPyConnection

    def load_sql(self, package: str, filename: str) -> str:
        return (files(package) / filename).read_text(encoding="utf-8")

    def execute_script(self, sql: str, params: dict | None = None) -> None:
        self.con.execute("BEGIN;")
        try:
            if params is not None:
                self.con.execute(sql, params)
            else:
                self.con.execute(sql)
            self.con.execute("COMMIT;")
        except Exception:
            self.con.execute("ROLLBACK;")
            raise
    
    def fetcharrow(self, sql: str, params: Any = None) -> pa.Table:
        if params is not None:
            return self.con.execute(sql, params).fetch_arrow_table()
        return self.con.execute(sql).fetch_arrow_table()
    
    def fetchdf(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        if params is not None:
            return self.con.execute(sql, params).fetchdf()
        return self.con.execute(sql).fetchdf()
    
    def count(self, relation: str) -> int:
        return int(self.con.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0])

