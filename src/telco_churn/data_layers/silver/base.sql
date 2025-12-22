-- silver.base: normalized, validated, and deduplicated customer table built from bronze.raw.

CREATE SCHEMA IF NOT EXISTS silver;

CREATE OR REPLACE TABLE silver.base AS
WITH raw AS (
  SELECT
    customerID, gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
    SeniorCitizen, tenure, MonthlyCharges, TotalCharges
  FROM bronze.raw
),
typed AS (
  SELECT
    trim(customerID)       AS customer_id,
    lower(trim(gender))           AS gender,
    lower(trim(Partner))          AS partner,
    lower(trim(Dependents))       AS dependents,
    lower(trim(PhoneService))     AS phone_service,
    lower(trim(MultipleLines))    AS multiple_lines,
    lower(trim(InternetService))  AS internet_service,
    lower(trim(OnlineSecurity))   AS online_security,
    lower(trim(OnlineBackup))     AS online_backup,
    lower(trim(DeviceProtection)) AS device_protection,
    lower(trim(TechSupport))      AS tech_support,
    lower(trim(StreamingTV))      AS streaming_tv,
    lower(trim(StreamingMovies))  AS streaming_movies,
    lower(trim(Contract))         AS contract,
    lower(trim(PaperlessBilling)) AS paperless_billing,
    lower(trim(PaymentMethod))    AS payment_method,
    TRY_CAST(SeniorCitizen  AS INTEGER) AS senior_citizen,
    TRY_CAST(tenure         AS INTEGER) AS tenure,
    TRY_CAST(MonthlyCharges AS DOUBLE)  AS monthly_charges,
    TRY_CAST(TotalCharges   AS DOUBLE)  AS total_charges
  FROM raw
),
validated AS (
  SELECT
    * REPLACE (
      CASE WHEN gender IN ('male','female') THEN gender ELSE NULL END AS gender,
      CASE WHEN partner IN ('yes','no') THEN partner ELSE NULL END AS partner,
      CASE WHEN dependents IN ('yes','no') THEN dependents ELSE NULL END AS dependents,
      CASE WHEN phone_service IN ('yes','no') THEN phone_service ELSE NULL END AS phone_service,
      CASE WHEN multiple_lines IN ('no phone service','yes','no') THEN multiple_lines ELSE NULL END AS multiple_lines,
      CASE WHEN internet_service IN ('dsl','fiber optic','no') THEN internet_service ELSE NULL END AS internet_service,
      CASE WHEN online_security IN ('yes','no','no internet service') THEN online_security ELSE NULL END AS online_security,
      CASE WHEN online_backup IN ('yes','no','no internet service') THEN online_backup ELSE NULL END AS online_backup,
      CASE WHEN device_protection IN ('yes','no','no internet service') THEN device_protection ELSE NULL END AS device_protection,
      CASE WHEN tech_support IN ('yes','no','no internet service') THEN tech_support ELSE NULL END AS tech_support,
      CASE WHEN streaming_tv IN ('yes','no','no internet service') THEN streaming_tv ELSE NULL END AS streaming_tv,
      CASE WHEN streaming_movies IN ('yes','no','no internet service') THEN streaming_movies ELSE NULL END AS streaming_movies,
      CASE WHEN contract IN ('month-to-month','one year','two year') THEN contract ELSE NULL END AS contract,
      CASE WHEN paperless_billing IN ('yes','no') THEN paperless_billing ELSE NULL END AS paperless_billing,
      CASE WHEN payment_method IN (
        'electronic check',
        'mailed check',
        'bank transfer (automatic)',
        'credit card (automatic)'
      ) THEN payment_method ELSE NULL END AS payment_method,
      CASE WHEN senior_citizen BETWEEN 0 AND 1 THEN senior_citizen ELSE NULL END AS senior_citizen,
      CASE WHEN tenure BETWEEN 0 AND 72 THEN tenure ELSE NULL END AS tenure,
      CASE WHEN monthly_charges BETWEEN 18.25 AND 118.75 THEN monthly_charges ELSE NULL END AS monthly_charges,
      CASE WHEN total_charges BETWEEN 18.8 AND 8684.8 THEN total_charges ELSE NULL END AS total_charges
    )
  FROM typed
),
deduped AS (
  SELECT *
  FROM (
    SELECT
      validated.*,
      ROW_NUMBER() OVER (
        PARTITION BY customer_id
        ORDER BY total_charges DESC NULLS LAST, tenure DESC NULLS LAST
      ) AS rn
    FROM validated
  )
  WHERE rn = 1
)
SELECT
  customer_id,
  gender,
  partner,
  dependents,
  phone_service,
  multiple_lines,
  internet_service,
  online_security,
  online_backup,
  device_protection,
  tech_support,
  streaming_tv,
  streaming_movies,
  contract,
  paperless_billing,
  payment_method,
  senior_citizen,
  tenure,
  monthly_charges,
  total_charges
FROM deduped;
