from datetime import datetime, timezone
import json
from pathlib import Path

def write_latest_pointer(*, reports_root: Path, batch_id: str) -> Path:
    latest = {
        "latest_batch_id": batch_id,
        "path": f"reports/batch_{batch_id}",
    }
    latest_path = reports_root / "latest.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(latest, f, indent=2)
    return latest_path
