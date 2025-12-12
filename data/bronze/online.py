import pandas as pd

URL = "https://huggingface.co/datasets/Carson-Shively/telco-churn/resolve/main/data/bronze/online_bronze.parquet"

bronze_py = pd.read_parquet(URL, engine="pyarrow")
