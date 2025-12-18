from __future__ import annotations
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from typing import Optional

def download_from_hf(repo_id: str, filename: str, revision: str = "main") -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )

def upload_parquet(
    local_path: str | Path,
    repo_id: str,
    hf_path: str,
    commit_message: str | None = None,
) -> None:
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(f"Local file not found: {p}")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(p),
        path_in_repo=hf_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message or f"Upload {hf_path}",
    )

def upload_bundle(
    bundle_dir: str | Path,
    *,
    repo_id: str,
    modeltype: str,
    revision: str = "main",
    commit_message: Optional[str] = None,
) -> None:
    p = Path(bundle_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {p}")

    files = [x for x in p.iterdir() if x.is_file()]
    if not files:
        raise ValueError(f"No files found in bundle directory: {p}")

    api = HfApi()
    api.upload_folder(
        folder_path=str(p),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        path_in_repo=f"candidates/{modeltype}",
        commit_message=commit_message or f"Upload candidates/{modeltype} bundle",
    )
