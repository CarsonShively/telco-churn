from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def preprocessor() -> ColumnTransformer:
    ohe_cols = [
        "gender_id", "multiple_lines_id", "internet_service_id",
        "online_security_id", "online_backup_id", "device_protection_id",
        "tech_support_id", "streaming_tv_id", "streaming_movies_id",
        "payment_method_id", "tenure_bucket_id", "contract_term_months",
    ]

    scaler_cols = [
        "tenure", "monthly_charges", "total_charges",
        "charges_per_month", "charges_ratio", "addon_service_ratio",
        "num_services", "num_addon_services",
    ]

    passthrough_cols = [
        "senior_citizen", "partner_id", "dependents_id", "phone_service_id",
        "paperless_billing_id", "short_tenure", "long_tenure",
        "is_month_to_month", "has_fiber", "no_internet",
        "high_monthly_charges", "security_bundle", "tech_bundle",
    ]

    ohe_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    scaler_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    bin_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", scaler_pipeline, scaler_cols),
            ("cat", ohe_pipeline, ohe_cols),
            ("bin", bin_pipeline, passthrough_cols),
        ],
        remainder="drop",
    )
