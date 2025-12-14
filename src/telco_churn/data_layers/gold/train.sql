CREATE SCHEMA IF NOT EXISTS gold;

CREATE OR REPLACE TABLE gold.training AS
SELECT
  f.*,
  l.churn
FROM gold.features f
JOIN silver.labels l
  ON l.customer_id = f.customer_id;