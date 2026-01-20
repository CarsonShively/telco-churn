import dagster as dg
from pathlib import Path

class DuckDBResource(dg.ConfigurableResource):
    path: str

    def db_path(self) -> Path:
        return Path(self.path).expanduser().resolve()