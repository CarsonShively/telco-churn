from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def tree_preprocessor() -> ColumnTransformer:

    features = [
        "tenure", "monthly_charges", "total_charges",
        "charges_per_month", "charges_ratio", "addon_service_ratio",
        "num_services", "num_addon_services", "senior_citizen", "partner_id", "dependents_id", "phone_service_id",
        "paperless_billing_id", "short_tenure", "long_tenure",
        "is_month_to_month", "has_fiber", "no_internet",
        "high_monthly_charges", "security_bundle", "tech_bundle", "gender_id", "multiple_lines_id", "internet_service_id",
        "online_security_id", "online_backup_id", "device_protection_id",
        "tech_support_id", "streaming_tv_id", "streaming_movies_id",
        "payment_method_id", "tenure_bucket_id", "contract_term_months",
    ]

    return ColumnTransformer(
        transformers=[
            ("impute_missing_value", SimpleImputer(missing_values=-1, strategy="most_frequent"), features),
        ],
        remainder="drop",
    )
