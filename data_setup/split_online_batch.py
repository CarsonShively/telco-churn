import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from telco_churn.io.hf import download_dataset_hf, upload_dataset_hf
from telco_churn.config import REPO_ID, REVISION

RAW_PATH_IN_REPO = "data/raw/churn.csv"
TRAIN_PATH_IN_REPO = "data/bronze/train.parquet"
DEMO_PATH_IN_REPO = "data/bronze/demo.parquet"

def main():
    raw = download_dataset_hf(repo_id=REPO_ID, revision=REVISION, filename=RAW_PATH_IN_REPO)
    df = pd.read_csv(raw)

    train_df, demo_df = train_test_split(
        df,
        test_size=0.60,
        random_state=42,
        stratify=df["Churn"],
    )

    demo_df = demo_df.drop(columns=["Churn"])

    bronze_dir = Path("data") / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)

    train_file = bronze_dir / "train.parquet"
    demo_file  = bronze_dir / "demo.parquet"

    train_df.to_parquet(train_file, index=False)
    demo_df.to_parquet(demo_file, index=False)

    upload_dataset_hf(local_path=str(train_file), repo_id=REPO_ID, hf_path=TRAIN_PATH_IN_REPO, revision=REVISION)
    upload_dataset_hf(local_path=str(demo_file),  repo_id=REPO_ID, hf_path=DEMO_PATH_IN_REPO,  revision=REVISION)

if __name__ == "__main__":
    main()
