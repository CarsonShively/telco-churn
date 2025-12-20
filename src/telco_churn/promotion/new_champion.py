from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from huggingface_hub import HfApi


@dataclass(frozen=True)
class ChampionRef:
    run_id: str
    path_in_repo: str


def write_champion_json_local(
    ref: ChampionRef,
    *,
    out_path: str | Path = "champion.json",
) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    obj = {
        "run_id": ref.run_id,
        "path_in_repo": ref.path_in_repo,
    }

    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p

def upload_existing_champion_json(
    local_path: str | Path,
    *,
    repo_id: str,
    repo_type: str,
    revision: str = "main",
    path_in_repo: str = "champion.json",
) -> None:
    p = Path(local_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Local champion.json not found: {p}")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        commit_message=f"Update champion pointer from {p.name}",
    )
