import pandas as pd

URL = "https://huggingface.co/datasets/Carson-Shively/telco-churn/resolve/main/data/bronze/offline_bronze.parquet"

df = pd.read_parquet(URL, engine="pyarrow")
