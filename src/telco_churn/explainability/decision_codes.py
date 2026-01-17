DECISION_CODES = {
    # ────────────────
    # Tenure / lifecycle
    # ────────────────
    "num__tenure": "NEW_CUSTOMER_RISK",
    "cat__tenure_bucket_id_0.0": "NEW_CUSTOMER_RISK",
    "cat__tenure_bucket_id_1.0": "NEW_CUSTOMER_RISK",
    "cat__tenure_bucket_id_2.0": "NEW_CUSTOMER_RISK",
    "bin__short_tenure": "NEW_CUSTOMER_RISK",

    "cat__tenure_bucket_id_3.0": "USAGE_INTENSITY",
    "cat__tenure_bucket_id_4.0": "USAGE_INTENSITY",
    "bin__long_tenure": "USAGE_INTENSITY",

    # ────────────────
    # Pricing & cost pressure
    # ────────────────
    "num__monthly_charges": "PRICE_SENSITIVITY",
    "num__total_charges": "PRICE_SENSITIVITY",
    "num__charges_per_month": "PRICE_SENSITIVITY",
    "num__charges_ratio": "PRICE_SENSITIVITY",
    "bin__high_monthly_charges": "PRICE_SENSITIVITY",

    # ────────────────
    # Product usage / intensity
    # ────────────────
    "num__num_services": "USAGE_INTENSITY",
    "bin__phone_service_id": "USAGE_INTENSITY",
    "bin__has_fiber": "USAGE_INTENSITY",
    "bin__no_internet": "USAGE_INTENSITY",
    "cat__multiple_lines_id_0.0": "USAGE_INTENSITY",
    "cat__multiple_lines_id_1.0": "USAGE_INTENSITY",
    "cat__multiple_lines_id_2.0": "USAGE_INTENSITY",


    # ────────────────
    # Add-ons & bundling
    # ────────────────
    "num__addon_service_ratio": "ADDON_DEPENDENCE",
    "num__num_addon_services": "ADDON_DEPENDENCE",
    "bin__security_bundle": "ADDON_DEPENDENCE",
    "bin__tech_bundle": "ADDON_DEPENDENCE",

    # ────────────────
    # Support & service friction
    # ────────────────
    "cat__online_security_id_0.0": "SUPPORT_FRICTION",
    "cat__online_backup_id_0.0": "SUPPORT_FRICTION",
    "cat__device_protection_id_0.0": "SUPPORT_FRICTION",
    "cat__tech_support_id_0.0": "SUPPORT_FRICTION",

    "cat__online_security_id_2.0": "SUPPORT_FRICTION",
    "cat__online_backup_id_2.0": "SUPPORT_FRICTION",
    "cat__device_protection_id_2.0": "SUPPORT_FRICTION",
    "cat__tech_support_id_2.0": "SUPPORT_FRICTION",
    "cat__online_security_id_1.0": "SUPPORT_FRICTION",
    "cat__online_backup_id_1.0": "SUPPORT_FRICTION",
    "cat__device_protection_id_1.0": "SUPPORT_FRICTION",
    "cat__tech_support_id_1.0": "SUPPORT_FRICTION",


    # ────────────────
    # Service quality / entertainment
    # ────────────────
    "cat__internet_service_id_0.0": "SERVICE_QUALITY",
    "cat__internet_service_id_1.0": "SERVICE_QUALITY",
    "cat__internet_service_id_2.0": "SERVICE_QUALITY",
    "cat__streaming_tv_id_0.0": "SERVICE_QUALITY",
    "cat__streaming_movies_id_0.0": "SERVICE_QUALITY",
    "cat__streaming_tv_id_1.0": "SERVICE_QUALITY",
    "cat__streaming_tv_id_2.0": "SERVICE_QUALITY",
    "cat__streaming_movies_id_1.0": "SERVICE_QUALITY",
    "cat__streaming_movies_id_2.0": "SERVICE_QUALITY",


    # ────────────────
    # Contract & commitment
    # ────────────────
    "bin__is_month_to_month": "LOW_COMMITMENT",
    "cat__contract_term_months_1.0": "LOW_COMMITMENT",
    "cat__contract_term_months_12.0": "LOW_COMMITMENT",
    "cat__contract_term_months_24.0": "LOW_COMMITMENT",

    # ────────────────
    # Billing & payment behavior
    # ────────────────
    "cat__payment_method_id_0.0": "BILLING_PAYMENT_RISK",
    "cat__payment_method_id_1.0": "BILLING_PAYMENT_RISK",
    "cat__payment_method_id_2.0": "BILLING_PAYMENT_RISK",
    "cat__payment_method_id_3.0": "BILLING_PAYMENT_RISK",
    "bin__paperless_billing_id": "BILLING_PAYMENT_RISK",

    # ────────────────
    # Household / demographics
    # ────────────────
    "bin__partner_id": "HOUSEHOLD_CONTEXT",
    "bin__dependents_id": "HOUSEHOLD_CONTEXT",
    "bin__senior_citizen": "DEMOGRAPHIC_SIGNAL",
    "cat__gender_id_0.0": "DEMOGRAPHIC_SIGNAL",
    "cat__gender_id_1.0": "DEMOGRAPHIC_SIGNAL",
}
