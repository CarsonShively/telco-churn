from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from telco_churn.modeling.custom_transformers.to_category import ToCategory
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

def lgb_preprocessor() -> ColumnTransformer:
    cat_cols = [
        "gender_id", "multiple_lines_id", "internet_service_id",
        "online_security_id", "online_backup_id", "device_protection_id",
        "tech_support_id", "streaming_tv_id", "streaming_movies_id",
        "payment_method_id", "tenure_bucket_id", "contract_term_months",
    ]

    num_cols = [
        "tenure", "monthly_charges", "total_charges",
        "charges_per_month", "charges_ratio", "addon_service_ratio",
        "num_services", "num_addon_services",
    ]

    bin_cols = [
        "senior_citizen", "partner_id", "dependents_id", "phone_service_id",
        "paperless_billing_id", "short_tenure", "long_tenure",
        "is_month_to_month", "has_fiber", "no_internet",
        "high_monthly_charges", "security_bundle", "tech_bundle",
    ]


    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_category", ToCategory()),
    ])

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    bin_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
            ("bin", bin_pipe, bin_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
