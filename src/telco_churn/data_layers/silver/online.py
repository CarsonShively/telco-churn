import pyarrow as pa
import pyarrow.compute as pc

def _lower_trim(col: pa.ChunkedArray | pa.Array) -> pa.Array:
    return pc.utf8_lower(pc.utf8_trim_whitespace(col))

def _try_int(col) -> pa.Array:
    return pc.try_cast(col, pa.int64())

def _try_float(col) -> pa.Array:
    return pc.try_cast(col, pa.float64())

def _validate_in(col: pa.Array, allowed: list[str]) -> pa.Array:
    allowed_arr = pa.array(allowed, type=pa.string())
    ok = pc.is_in(col, value_set=allowed_arr)
    return pc.if_else(ok, col, pa.scalar(None, type=col.type))

def _validate_range(col: pa.Array, lo: float, hi: float) -> pa.Array:
    ok = pc.and_(pc.greater_equal(col, lo), pc.less_equal(col, hi))
    return pc.if_else(ok, col, pa.scalar(None, type=col.type))

def _first_row_per_key_sorted(table: pa.Table, key: str) -> pa.Table:
    key_arr = table[key].combine_chunks()

    prev = pa.concat_arrays([
        pa.array([None], type=key_arr.type),
        key_arr.slice(0, len(key_arr) - 1),
    ])

    keep = pc.or_(pc.is_null(prev), pc.not_equal(key_arr, prev))
    return table.filter(keep)


def build_silver_offline_from_bronze(bronze: pa.Table) -> pa.Table:
    b = bronze  

    customer_id      = _lower_trim(b["customerID"])
    gender           = _lower_trim(b["gender"])
    partner          = _lower_trim(b["Partner"])
    dependents       = _lower_trim(b["Dependents"])
    phone_service    = _lower_trim(b["PhoneService"])
    multiple_lines   = _lower_trim(b["MultipleLines"])
    internet_service = _lower_trim(b["InternetService"])
    online_security  = _lower_trim(b["OnlineSecurity"])
    online_backup    = _lower_trim(b["OnlineBackup"])
    device_protect   = _lower_trim(b["DeviceProtection"])
    tech_support     = _lower_trim(b["TechSupport"])
    streaming_tv     = _lower_trim(b["StreamingTV"])
    streaming_movies = _lower_trim(b["StreamingMovies"])
    contract         = _lower_trim(b["Contract"])
    paperless_bill   = _lower_trim(b["PaperlessBilling"])
    payment_method   = _lower_trim(b["PaymentMethod"])
    churn            = _lower_trim(b["Churn"])

    senior_citizen   = _try_int(b["SeniorCitizen"])
    tenure           = _try_int(b["tenure"])
    monthly_charges  = _try_float(b["MonthlyCharges"])
    total_charges    = _try_float(b["TotalCharges"])

    typed = pa.table({
        "customer_id": customer_id,
        "gender": gender,
        "partner": partner,
        "dependents": dependents,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protect,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_bill,
        "payment_method": payment_method,
        "churn": churn,
        "senior_citizen": senior_citizen,
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
    })

    validated = typed.set_column(
        typed.schema.get_field_index("gender"),
        "gender",
        _validate_in(typed["gender"], ["male", "female"]),
    ).set_column(
        typed.schema.get_field_index("partner"),
        "partner",
        _validate_in(typed["partner"], ["yes", "no"]),
    ).set_column(
        typed.schema.get_field_index("dependents"),
        "dependents",
        _validate_in(typed["dependents"], ["yes", "no"]),
    ).set_column(
        typed.schema.get_field_index("phone_service"),
        "phone_service",
        _validate_in(typed["phone_service"], ["yes", "no"]),
    ).set_column(
        typed.schema.get_field_index("multiple_lines"),
        "multiple_lines",
        _validate_in(typed["multiple_lines"], ["no phone service", "yes", "no"]),
    ).set_column(
        typed.schema.get_field_index("internet_service"),
        "internet_service",
        _validate_in(typed["internet_service"], ["dsl", "fiber optic", "no"]),
    ).set_column(
        typed.schema.get_field_index("online_security"),
        "online_security",
        _validate_in(typed["online_security"], ["yes", "no", "no internet service"]),
    ).set_column(
        typed.schema.get_field_index("online_backup"),
        "online_backup",
        _validate_in(typed["online_backup"], ["yes", "no", "no internet service"]),
    ).set_column(
        typed.schema.get_field_index("device_protection"),
        "device_protection",
        _validate_in(typed["device_protection"], ["yes", "no", "no internet service"]),
    ).set_column(
        typed.schema.get_field_index("tech_support"),
        "tech_support",
        _validate_in(typed["tech_support"], ["yes", "no", "no internet service"]),
    ).set_column(
        typed.schema.get_field_index("streaming_tv"),
        "streaming_tv",
        _validate_in(typed["streaming_tv"], ["yes", "no", "no internet service"]),
    ).set_column(
        typed.schema.get_field_index("streaming_movies"),
        "streaming_movies",
        _validate_in(typed["streaming_movies"], ["yes", "no", "no internet service"]),
    ).set_column(
        typed.schema.get_field_index("contract"),
        "contract",
        _validate_in(typed["contract"], ["month-to-month", "one year", "two year"]),
    ).set_column(
        typed.schema.get_field_index("paperless_billing"),
        "paperless_billing",
        _validate_in(typed["paperless_billing"], ["yes", "no"]),
    ).set_column(
        typed.schema.get_field_index("payment_method"),
        "payment_method",
        _validate_in(
            typed["payment_method"],
            [
                "electronic check",
                "mailed check",
                "bank transfer (automatic)",
                "credit card (automatic)",
            ],
        ),
    ).set_column(
        typed.schema.get_field_index("churn"),
        "churn",
        _validate_in(typed["churn"], ["yes", "no"]),
    ).set_column(
        typed.schema.get_field_index("senior_citizen"),
        "senior_citizen",
        _validate_range(typed["senior_citizen"], 0, 1),
    ).set_column(
        typed.schema.get_field_index("tenure"),
        "tenure",
        _validate_range(typed["tenure"], 0, 72),
    ).set_column(
        typed.schema.get_field_index("monthly_charges"),
        "monthly_charges",
        _validate_range(typed["monthly_charges"], 18.25, 118.75),
    ).set_column(
        typed.schema.get_field_index("total_charges"),
        "total_charges",
        _validate_range(typed["total_charges"], 18.8, 8684.8),
    )

    sorted_tbl = validated.sort_by([
        ("customer_id", "ascending"),
        ("total_charges", "descending"),
        ("tenure", "descending"),
    ])
    deduped = _first_row_per_key_sorted(sorted_tbl, "customer_id")

    return deduped
