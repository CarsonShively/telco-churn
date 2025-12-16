import os
from dataclasses import dataclass

from telco_churn.io.hf import download_from_hf
from telco_churn.modeling.dataframe.train_to_df import parquet_to_df

@dataclass(frozen=True)
class PipelineConfig:
    repo_id: str = os.getenv("TELCO_REPO_ID", "Carson-Shively/telco-churn")
    revision: str = os.getenv("TELCO_REVISION", "main")
    train_hf_path: str = os.getenv("TELCO_TRAIN_HF_PATH", "data/gold/train.parquet")

def main(cfg: PipelineConfig = PipelineConfig()) -> None:
    local_path = download_from_hf(
        repo_id=cfg.repo_id,
        filename=cfg.train_hf_path,
        revision=cfg.revision,
    )
    df = parquet_to_df(local_path)
    
    X = 


if __name__ == "__main__":
    main()
