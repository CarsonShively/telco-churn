from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


@dataclass(frozen=True)
class ChampionRef:
    run_id: str
    path_in_repo: str


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def try_load_champion_ref(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    champion_path: str = "champion.json",
) -> Optional[ChampionRef]:
    try:
        local = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=champion_path,
                revision=revision,
            )
        )
    except EntryNotFoundError:
        return None

    data = _read_json(local)
    run_id = data.get("run_id")
    if not run_id:
        return None

    path_in_repo = data.get("path_in_repo") or f"runs/{run_id}"
    return ChampionRef(run_id=run_id, path_in_repo=path_in_repo)

def load_champion_metrics(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    ref: ChampionRef,
) -> Dict[str, Any]:
    metrics_file = f"{ref.path_in_repo}/metrics.json"

    local = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        filename=metrics_file,
    )
    return json.loads(Path(local).read_text(encoding="utf-8"))

