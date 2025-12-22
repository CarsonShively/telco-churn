from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

@dataclass(frozen=True)
class ChampionPointer:
    run_id: str
    path_in_repo: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChampionPointer":
        run_id = d.get("run_id")
        path_in_repo = d.get("path_in_repo")
        if not run_id or not path_in_repo:
            raise ValueError("champion.json must contain 'run_id' and 'path_in_repo'")
        return cls(run_id=str(run_id), path_in_repo=str(path_in_repo))

    def model_path(self, *, filename: str = "model.joblib") -> str:
        return f"{self.path_in_repo}/{filename}".replace("//", "/")

def read_json(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str,
) -> dict[str, Any]:
    """
    Download a JSON file from Hugging Face Hub and return it as a dict.

    Raises:
      - EntryNotFoundError if the file doesn't exist at that path/revision.
      - json.JSONDecodeError if the file is not valid JSON.
    """
    local_file = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        filename=path_in_repo,
    )
    with open(local_file, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_champion_pointer(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str = "champion.json",
) -> Optional["ChampionPointer"]:
    try:
        data = read_json(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            path_in_repo=path_in_repo,
        )
    except EntryNotFoundError:
        return None
    return ChampionPointer.from_dict(data)


def load_model_from_champion_pointer(
    champion: "ChampionPointer",
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    model_filename: str = "model.joblib",
) -> Any:
    local_file = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        filename=champion.model_path(filename=model_filename),
    )
    return joblib.load(Path(local_file))
