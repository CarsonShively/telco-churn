from huggingface_hub import hf_hub_download

def download_from_hf(repo_id: str, filename: str, revision: str = "main") -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )