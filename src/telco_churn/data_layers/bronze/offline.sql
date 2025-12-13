LOAD httpfs;

CREATE SCHEMA IF NOT EXISTS bronze;

CREATE OR REPLACE VIEW bronze.offline AS
SELECT *
FROM read_parquet(
  'https://huggingface.co/datasets/Carson-Shively/telco-churn/resolve/main/data/bronze/offline.parquet'
);