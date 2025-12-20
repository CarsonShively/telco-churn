from __future__ import annotations
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi, snapshot_download
from typing import Optional, Any, Dict, List
import json

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
    run_id: str,
    revision: str = "main",
    commit_message: Optional[str] = None,
    ensure_new: bool = True,
) -> str:
    p = Path(bundle_dir)

    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {p}")

    inferred = p.name
    if run_id != inferred:
        raise ValueError(f"run_id mismatch: arg={run_id} folder={inferred}")

    path_in_repo = f"runs/{run_id}"
    api = HfApi()

    if ensure_new:
        existing_files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
        )
        prefix = path_in_repo + "/"
        if any(f.startswith(prefix) for f in existing_files):
            raise FileExistsError(
                f"Remote run folder already exists: {path_in_repo} "
                f"(choose a new run_id or disable ensure_new)"
            )

    api.upload_folder(
        folder_path=str(p),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        path_in_repo=path_in_repo,
        commit_message=commit_message or f"Upload run {run_id}",
    )

    return path_in_repo



def fetch_all_candidate_metrics(
    *,
    repo_id: str,
    repo_type: str = "model",
    revision: str = "main",
    cache_dir: str = "hf_cache",
) -> List[Dict[str, Any]]:

    root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=["candidates/**/metrics.json"],
        )
    )

    rows: List[Dict[str, Any]] = []
    for metrics_path in root.rglob("candidates/**/metrics.json"):
        if "candidates" not in metrics_path.parts:
            continue

        parts = metrics_path.parts
        i = parts.index("candidates")
        model_type: Optional[str] = parts[i + 1] if len(parts) > i + 1 else None
        run_id: Optional[str] = parts[i + 2] if len(parts) > i + 2 else None

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        rows.append(
            {
                "model_type": metrics.get("model_type", model_type),
                "run_id": metrics.get("run_id", run_id),
                "metrics_path": str(metrics_path),
                "metrics": metrics,
            }
        )

    return rows

def fetch_champion_pointer(
    *,
    repo_id: str,
    repo_type: str = "model",
    revision: str = "main",
    filename: str = "champion.json",
    cache_dir: str = "hf_cache",
) -> Optional[Dict[str, Any]]:
    try:
        p = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            filename=filename,
            cache_dir=cache_dir,
        )
    except Exception:
        return None

    return json.loads(Path(p).read_text(encoding="utf-8"))