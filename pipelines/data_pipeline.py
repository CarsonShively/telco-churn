import duckdb
from telco_churn.data_layers.bronze.online import read_parquet_arrow
from telco_churn.io.hf import download_from_hf
from telco_churn.executor.executor_sql import SQLExecutor

repo_id = "Carson-Shively/telco-churn"
filename = "data/bronze/offline.parquet"
BRONZE_SQL_PKG = "telco_churn.data.bronze"
BRONZE_SQL_FILE = "offline.sql"

con = duckdb.connect()
ex = SQLExecutor(con)

local_path = download_from_hf(repo_id=repo_id, filename=filename)

sql = ex.load_sql(BRONZE_SQL_PKG, BRONZE_SQL_FILE)
ex.execute_script(sql, [local_path])

bronze_table = read_parquet_arrow(local_path)