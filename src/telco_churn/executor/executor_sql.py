from dataclasses import dataclass
from importlib.resources import files
import duckdb
from typing import Any

Params = tuple[Any, ...] | list[Any] | dict[str, Any] | None

@dataclass
class SQLExecutor:
    con: duckdb.DuckDBPyConnection

    def load_sql(self, package: str, filename: str) -> str:
        return (files(package) / filename).read_text(encoding="utf-8")

    def execute(self, sql: str, params: Params = None) -> None:
        if params is not None:
            self.con.execute(sql, params)
        else:
            self.con.execute(sql)

    def execute_script(self, sql: str, params: Params = None) -> None:
        self.con.execute("BEGIN;")
        try:
            self.execute(sql, params)
            self.con.execute("COMMIT;")
        except Exception:
            self.con.execute("ROLLBACK;")
            raise

    def write_parquet(self, select_sql: str, out_path: str) -> None:
        stmt = f"COPY ({select_sql}) TO '{out_path}' (FORMAT PARQUET)"
        self.execute(stmt)