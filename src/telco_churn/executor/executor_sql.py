from dataclasses import dataclass
from importlib.resources import files
import duckdb

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
    
    def write_parquet(self, select_sql: str, out_path: str) -> None:
        stmt = f"COPY ({select_sql}) TO '{out_path}' (FORMAT PARQUET)"
        self.execute(stmt)
    
    def count(self, relation: str) -> int:
        return int(self.con.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0])