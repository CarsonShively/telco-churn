-- silver.labels: customer-level churn label (0/1) keyed by customer_id from bronze.raw.

CREATE SCHEMA IF NOT EXISTS silver;

CREATE OR REPLACE TABLE silver.labels AS
SELECT
  trim(customerID) AS customer_id,
  CASE
    WHEN lower(trim(Churn)) = 'yes' THEN 1
    WHEN lower(trim(Churn)) = 'no'  THEN 0
    ELSE NULL
  END AS churn
FROM bronze.train;