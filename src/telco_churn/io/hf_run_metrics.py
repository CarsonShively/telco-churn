from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import HfApi, hf_hub_download


@dataclass(frozen=True)
class RunRow:
    run_id: str
    model_type: Optional[str]
    metrics: dict[str, Any]
    metrics_path: str
    error: Optional[str] = None


def extract_run_id_from_path(path_in_repo: str) -> Optional[str]:
    parts = path_in_repo.split("/")
    if len(parts) >= 3 and parts[0] == "runs" and parts[-1] == "metrics.json":
        return parts[1]
    return None


def _read_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fetch_all_run_metrics(*, repo_id: str, revision: str = "main") -> list[RunRow]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)

    metrics_paths = [f for f in files if f.startswith("runs/") and f.endswith("/metrics.json")]

    rows: list[RunRow] = []
    for mp in metrics_paths:
        run_id = extract_run_id_from_path(mp)
        if run_id is None:
            continue

        try:
            local_metrics = hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                filename=mp,
            )
            metrics = _read_json(local_metrics)
            model_type = metrics.get("model_type")

            rows.append(
                RunRow(
                    run_id=run_id,
                    model_type=model_type,
                    metrics=metrics,
                    metrics_path=mp,
                )
            )
        except Exception as e:
            rows.append(
                RunRow(
                    run_id=run_id,
                    model_type=None,
                    metrics={},
                    metrics_path=mp,
                    error=f"metrics_download_or_parse_failed: {type(e).__name__}: {e}",
                )
            )

    return rows
