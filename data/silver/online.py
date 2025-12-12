import pandas as pd

ALLOWED = {
    "gender": {"male", "female"},
    "partner": {"yes", "no"},
    "dependents": {"yes", "no"},
    "phone_service": {"yes", "no"},
    "multiple_lines": {"no phone service", "yes", "no"},
    "internet_service": {"dsl", "fiber optic", "no"},
    "online_security": {"yes", "no", "no internet service"},
    "online_backup": {"yes", "no", "no internet service"},
    "device_protection": {"yes", "no", "no internet service"},
    "tech_support": {"yes", "no", "no internet service"},
    "streaming_tv": {"yes", "no", "no internet service"},
    "streaming_movies": {"yes", "no", "no internet service"},
    "contract": {"month-to-month", "one year", "two year"},
    "paperless_billing": {"yes", "no"},
    "payment_method": {
        "electronic check",
        "mailed check",
        "bank transfer (automatic)",
        "credit card (automatic)",
    },
}

RANGES = {
    "senior_citizen": (0, 1),
    "tenure": (0, 72),
    "monthly_charges": (18.25, 118.75),
    "total_charges": (18.8, 8684.8),
}

RENAME_MAP = {
    "customerID": "customer_id",
    "SeniorCitizen": "senior_citizen",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "Partner": "partner",
    "Dependents": "dependents",
    "gender": "gender",
    "tenure": "tenure",
}

FEATURE_COLS_OUT = [
    "customer_id",
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
]

def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()

def _validate_allowed(s: pd.Series, allowed: set[str]) -> pd.Series:
    return s.where(s.isin(allowed), pd.NA)

def _validate_range(num: pd.Series, lo: float, hi: float) -> pd.Series:
    return num.where(num.between(lo, hi), pd.NA)

def make_silver_online(bronze_df: pd.DataFrame) -> pd.DataFrame:
    df = bronze_df.copy()

    df = df.rename(columns=RENAME_MAP)

    cat_cols = [c for c in ALLOWED.keys() if c in df.columns]
    for c in cat_cols:
        df[c] = _norm_str(df[c])

    if "customer_id" not in df.columns:
        raise KeyError("Missing required column: customerID/customer_id")
    df["customer_id"] = _norm_str(df["customer_id"])

    if "senior_citizen" in df.columns:
        df["senior_citizen"] = pd.to_numeric(df["senior_citizen"], errors="coerce").astype("Int64")
    else:
        df["senior_citizen"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").astype("Int64")
    else:
        df["tenure"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    for c in ["monthly_charges", "total_charges"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA

    for c, allowed in ALLOWED.items():
        if c in df.columns:
            df[c] = _validate_allowed(df[c], allowed)

    for c, (lo, hi) in RANGES.items():
        if c in df.columns:
            df[c] = _validate_range(df[c], lo, hi)

    df = df.sort_values(
        by=["customer_id", "total_charges", "tenure"],
        ascending=[True, False, False],
        na_position="last",
        kind="mergesort",
    )
    df = df.drop_duplicates(subset=["customer_id"], keep="first")

    for c in FEATURE_COLS_OUT:
        if c not in df.columns:
            df[c] = pd.NA

    return df[FEATURE_COLS_OUT].reset_index(drop=True)
