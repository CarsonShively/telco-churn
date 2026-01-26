"""
Per-run batch context.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import dagster as dg
from telco_churn.paths import REPO_ROOT

@dataclass(frozen=True)
class BatchRunContext:
    batch_id: str
    reports_root: Path
    batch_root: Path
    hf_batch_path: str
    scored_path: Path
    actions_path: Path
    summary_path: Path

class BatchContextResource(dg.ConfigurableResource):
    repo_root: str = "."
    reports_dirname: str = "reports"

    def get(self) -> BatchRunContext:
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")

        reports_root = REPO_ROOT / self.reports_dirname
        batch_root = reports_root / f"batch_{batch_id}"
        batch_root.mkdir(parents=True, exist_ok=True)

        scored_path = batch_root / "scored.parquet"
        actions_path = batch_root / "actions.parquet"
        summary_path = batch_root / "summary.json"
        hf_batch_path = f"reports/batch_{batch_id}"

        return BatchRunContext(
            batch_id=batch_id,
            reports_root=reports_root,
            batch_root=batch_root,
            hf_batch_path=hf_batch_path,
            scored_path=scored_path,
            actions_path=actions_path,
            summary_path=summary_path,
        )
