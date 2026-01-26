"""
Resolves the DuckDB database path.
"""

import dagster as dg
from pathlib import Path

class DuckDBResource(dg.ConfigurableResource):
    path: str

    def db_path(self) -> Path:
        p = Path(self.path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p.resolve()
