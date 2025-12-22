-- gold.train: final training dataset (gold.features joined with silver.labels on customer_id).

CREATE SCHEMA IF NOT EXISTS gold;

CREATE OR REPLACE TABLE gold.train AS
SELECT
  f.*,
  l.churn
FROM gold.features f
JOIN silver.labels l
  ON l.customer_id = f.customer_id;