-- Feature engineering layer built from silver.base; defines helper macros and encodes categories to numeric features.

CREATE SCHEMA IF NOT EXISTS gold;

CREATE OR REPLACE MACRO flag(expr) AS (
  CASE WHEN (expr) THEN 1 ELSE 0 END
);

CREATE OR REPLACE MACRO safe_div(num, den) AS (
  num / NULLIF(den, 0)
);

CREATE OR REPLACE MACRO gender_id(g) AS (
  CASE
    WHEN g = 'female' THEN 0
    WHEN g = 'male'   THEN 1
    ELSE NULL
  END
);

CREATE OR REPLACE MACRO yn01(x) AS (
  CASE
    WHEN x = 'yes' THEN 1
    WHEN x = 'no'  THEN 0
    ELSE NULL
  END
);

CREATE OR REPLACE MACRO internet_addon_id(x) AS (
  CASE
    WHEN x = 'no internet service' THEN 0
    WHEN x = 'no'                  THEN 1
    WHEN x = 'yes'                 THEN 2
    ELSE NULL
  END
);

CREATE OR REPLACE MACRO multiple_lines_id(x) AS (
  CASE
    WHEN x = 'no phone service' THEN 0
    WHEN x = 'no'               THEN 1
    WHEN x = 'yes'              THEN 2
    ELSE NULL
  END
);

CREATE OR REPLACE MACRO internet_service_id(x) AS (
  CASE
    WHEN x = 'no'          THEN 0
    WHEN x = 'dsl'         THEN 1
    WHEN x = 'fiber optic' THEN 2
    ELSE NULL
  END
);

CREATE OR REPLACE MACRO contract_term_months(c) AS (
  CASE
    WHEN c = 'month-to-month' THEN 1
    WHEN c = 'one year'       THEN 12
    WHEN c = 'two year'       THEN 24
    ELSE NULL
  END
);

CREATE OR REPLACE MACRO payment_method_id(p) AS (
  CASE
    WHEN p = 'electronic check'          THEN 0
    WHEN p = 'mailed check'              THEN 1
    WHEN p = 'bank transfer (automatic)' THEN 2
    WHEN p = 'credit card (automatic)'   THEN 3
    ELSE NULL
  END
);

CREATE OR REPLACE MACRO tenure_bucket_id(t) AS (
  CASE
    WHEN t IS NULL  THEN NULL
    WHEN t < 6      THEN 0
    WHEN t < 12     THEN 1
    WHEN t < 24     THEN 2
    WHEN t < 48     THEN 3
    ELSE 4
  END
);

CREATE OR REPLACE TABLE {features_table} AS
SELECT
    customer_id,

    senior_citizen,
    tenure,
    monthly_charges,
    total_charges,

    gender_id(gender)                     AS gender_id,
    yn01(partner)                         AS partner_id,
    yn01(dependents)                      AS dependents_id,
    yn01(phone_service)                   AS phone_service_id,
    multiple_lines_id(multiple_lines)     AS multiple_lines_id,
    internet_service_id(internet_service) AS internet_service_id,
    internet_addon_id(online_security)      AS online_security_id,
    internet_addon_id(online_backup)        AS online_backup_id,
    internet_addon_id(device_protection)    AS device_protection_id,
    internet_addon_id(tech_support)         AS tech_support_id,
    internet_addon_id(streaming_tv)         AS streaming_tv_id,
    internet_addon_id(streaming_movies)     AS streaming_movies_id,
    contract_term_months(contract)        AS contract_term_months,
    yn01(paperless_billing)               AS paperless_billing_id,
    payment_method_id(payment_method)     AS payment_method_id,

    (
      flag(phone_service   = 'yes') +
      flag(multiple_lines  = 'yes') +
      flag(online_security = 'yes') +
      flag(online_backup   = 'yes') +
      flag(device_protection = 'yes') +
      flag(tech_support    = 'yes') +
      flag(streaming_tv    = 'yes') +
      flag(streaming_movies = 'yes')
    ) AS num_services,

    (
      flag(online_security = 'yes') +
      flag(online_backup   = 'yes') +
      flag(device_protection = 'yes') +
      flag(tech_support    = 'yes') +
      flag(streaming_tv    = 'yes') +
      flag(streaming_movies = 'yes')
    ) AS num_addon_services,

    flag(online_security = 'yes' OR device_protection = 'yes') AS security_bundle,
    flag(tech_support    = 'yes')                              AS tech_bundle,

    safe_div(total_charges, tenure)          AS charges_per_month,
    safe_div(total_charges, monthly_charges) AS charges_ratio,
    safe_div(
      (
        flag(online_security = 'yes') +
        flag(online_backup   = 'yes') +
        flag(device_protection = 'yes') +
        flag(tech_support    = 'yes') +
        flag(streaming_tv    = 'yes') +
        flag(streaming_movies = 'yes')
      ),
      6
    ) AS addon_service_ratio,

    tenure_bucket_id(tenure)               AS tenure_bucket_id,
    flag(tenure < 6)                       AS short_tenure,
    flag(tenure > 24)                      AS long_tenure,
    flag(contract = 'month-to-month')      AS is_month_to_month,
    flag(internet_service = 'fiber optic') AS has_fiber,
    flag(internet_service = 'no')          AS no_internet,
    flag(monthly_charges > 80)             AS high_monthly_charges

FROM {base_table};