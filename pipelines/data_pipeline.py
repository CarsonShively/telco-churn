import duckdb
from telco_churn.data_layers.bronze.online import read_parquet_arrow
from telco_churn.io.hf import download_from_hf
from telco_churn.executor.executor_sql import SQLExecutor
from telco_churn.data_layers.silver.online import build_silver_offline_from_bronze
from telco_churn.parity.parity_test import assert_silver_parity

repo_id = "Carson-Shively/telco-churn"
filename = "data/bronze/offline.parquet"
BRONZE_SQL_PKG = "telco_churn.data_layers.bronze"
BRONZE_SQL_FILE = "offline.sql"

SILVER_SQL_PKG = "telco_churn.data_layers.silver"
SILVER_SQL_FILE = "offline.sql"

con = duckdb.connect()
ex = SQLExecutor(con)

local_path = download_from_hf(repo_id=repo_id, filename=filename)

bronze_sql = ex.load_sql(BRONZE_SQL_PKG, BRONZE_SQL_FILE)
ex.execute_script(bronze_sql, [local_path])

bronze_online = read_parquet_arrow(local_path)

silver_sql = ex.load_sql(SILVER_SQL_PKG, SILVER_SQL_FILE)
ex.execute_script(silver_sql)
silver_offline = ex.fetcharrow("SELECT * FROM silver.offline")

silver_online = build_silver_offline_from_bronze(bronze_online)

assert_silver_parity(silver_offline, silver_online)