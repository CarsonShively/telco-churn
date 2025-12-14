from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi


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
