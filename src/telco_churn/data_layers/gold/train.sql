-- Final training dataset (gold.features joined with silver.labels on customer_id).

CREATE SCHEMA IF NOT EXISTS gold;

CREATE OR REPLACE TABLE gold.join_train AS
SELECT
  f.*,
  l.churn
FROM gold.train_features f
JOIN silver.labels l
  ON l.customer_id = f.customer_id;