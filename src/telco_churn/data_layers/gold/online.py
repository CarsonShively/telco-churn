import pyarrow as pa
import pyarrow.compute as pc


def _flag(expr: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return pc.if_else(pc.fill_null(expr, False), pa.scalar(1, pa.int8()), pa.scalar(0, pa.int8()))


def _safe_div(num, den):
    den_is_zero = pc.equal(den, 0)
    bad = pc.or_(pc.is_null(den), den_is_zero)
    return pc.if_else(bad, pa.nulls(pc.length(num), pa.float64()), pc.divide(num, den))


def _gender_id(g):
    return pc.case_when(
        [
            (pc.equal(g, "female"), pa.scalar(0, pa.int8())),
            (pc.equal(g, "male"), pa.scalar(1, pa.int8())),
        ],
        else_=pa.nulls(pc.length(g), pa.int8()),
    )


def _yn01(x):
    return pc.case_when(
        [
            (pc.equal(x, "yes"), pa.scalar(1, pa.int8())),
            (pc.equal(x, "no"), pa.scalar(0, pa.int8())),
        ],
        else_=pa.nulls(pc.length(x), pa.int8()),
    )


def _internet_addon_id(x):
    return pc.case_when(
        [
            (pc.equal(x, "no internet service"), pa.scalar(0, pa.int8())),
            (pc.equal(x, "no"), pa.scalar(1, pa.int8())),
            (pc.equal(x, "yes"), pa.scalar(2, pa.int8())),
        ],
        else_=pa.nulls(pc.length(x), pa.int8()),
    )


def _multiple_lines_id(x):
    return pc.case_when(
        [
            (pc.equal(x, "no phone service"), pa.scalar(0, pa.int8())),
            (pc.equal(x, "no"), pa.scalar(1, pa.int8())),
            (pc.equal(x, "yes"), pa.scalar(2, pa.int8())),
        ],
        else_=pa.nulls(pc.length(x), pa.int8()),
    )


def _internet_service_id(x):
    return pc.case_when(
        [
            (pc.equal(x, "no"), pa.scalar(0, pa.int8())),
            (pc.equal(x, "dsl"), pa.scalar(1, pa.int8())),
            (pc.equal(x, "fiber optic"), pa.scalar(2, pa.int8())),
        ],
        else_=pa.nulls(pc.length(x), pa.int8()),
    )


def _contract_term_months(c):
    return pc.case_when(
        [
            (pc.equal(c, "month-to-month"), pa.scalar(1, pa.int16())),
            (pc.equal(c, "one year"), pa.scalar(12, pa.int16())),
            (pc.equal(c, "two year"), pa.scalar(24, pa.int16())),
        ],
        else_=pa.nulls(pc.length(c), pa.int16()),
    )


def _payment_method_id(p):
    return pc.case_when(
        [
            (pc.equal(p, "electronic check"), pa.scalar(0, pa.int8())),
            (pc.equal(p, "mailed check"), pa.scalar(1, pa.int8())),
            (pc.equal(p, "bank transfer (automatic)"), pa.scalar(2, pa.int8())),
            (pc.equal(p, "credit card (automatic)"), pa.scalar(3, pa.int8())),
        ],
        else_=pa.nulls(pc.length(p), pa.int8()),
    )


def _tenure_bucket_id(t):
    n = pc.length(t)
    null_out = pa.nulls(n, pa.int8())

    is_null = pc.is_null(t)
    out = pc.if_else(is_null, null_out, pa.scalar(4, pa.int8()))
    out = pc.if_else(pc.and_(pc.invert(is_null), pc.less(t, 48)), pa.scalar(3, pa.int8()), out)
    out = pc.if_else(pc.and_(pc.invert(is_null), pc.less(t, 24)), pa.scalar(2, pa.int8()), out)
    out = pc.if_else(pc.and_(pc.invert(is_null), pc.less(t, 12)), pa.scalar(1, pa.int8()), out)
    out = pc.if_else(pc.and_(pc.invert(is_null), pc.less(t, 6)), pa.scalar(0, pa.int8()), out)
    return out


def build_gold_offline_from_silver(silver: pa.Table) -> pa.Table:
    churn = silver["churn"]
    customer_id = silver["customer_id"]

    senior_citizen = silver["senior_citizen"]
    tenure = silver["tenure"]
    monthly_charges = silver["monthly_charges"]
    total_charges = silver["total_charges"]

    gender = silver["gender"]
    partner = silver["partner"]
    dependents = silver["dependents"]
    phone_service = silver["phone_service"]
    multiple_lines = silver["multiple_lines"]
    internet_service = silver["internet_service"]
    online_security = silver["online_security"]
    online_backup = silver["online_backup"]
    device_protection = silver["device_protection"]
    tech_support = silver["tech_support"]
    streaming_tv = silver["streaming_tv"]
    streaming_movies = silver["streaming_movies"]
    contract = silver["contract"]
    paperless_billing = silver["paperless_billing"]
    payment_method = silver["payment_method"]

    churn01 = _yn01(churn)
    gender01 = _gender_id(gender)
    partner01 = _yn01(partner)
    dependents01 = _yn01(dependents)
    phone_service01 = _yn01(phone_service)
    multiple_lines01 = _multiple_lines_id(multiple_lines)
    internet_service01 = _internet_service_id(internet_service)

    online_security_id = _internet_addon_id(online_security)
    online_backup_id = _internet_addon_id(online_backup)
    device_protection_id = _internet_addon_id(device_protection)
    tech_support_id = _internet_addon_id(tech_support)
    streaming_tv_id = _internet_addon_id(streaming_tv)
    streaming_movies_id = _internet_addon_id(streaming_movies)

    contract_months = _contract_term_months(contract)
    paperless01 = _yn01(paperless_billing)
    payment_id = _payment_method_id(payment_method)

    f_phone = _flag(pc.equal(phone_service, "yes"))
    f_mult = _flag(pc.equal(multiple_lines, "yes"))
    f_sec = _flag(pc.equal(online_security, "yes"))
    f_backup = _flag(pc.equal(online_backup, "yes"))
    f_dev = _flag(pc.equal(device_protection, "yes"))
    f_tech = _flag(pc.equal(tech_support, "yes"))
    f_tv = _flag(pc.equal(streaming_tv, "yes"))
    f_movies = _flag(pc.equal(streaming_movies, "yes"))

    num_services = pc.add(pc.add(pc.add(f_phone, f_mult), pc.add(f_sec, f_backup)),
                          pc.add(pc.add(f_dev, f_tech), pc.add(f_tv, f_movies)))

    num_addon_services = pc.add(pc.add(f_sec, f_backup),
                               pc.add(pc.add(f_dev, f_tech), pc.add(f_tv, f_movies)))

    security_bundle = _flag(pc.or_(pc.equal(online_security, "yes"), pc.equal(device_protection, "yes")))
    tech_bundle = _flag(pc.equal(tech_support, "yes"))

    charges_per_month = _safe_div(total_charges, tenure)
    charges_ratio = _safe_div(total_charges, monthly_charges)
    addon_service_ratio = _safe_div(num_addon_services, pa.scalar(6, pa.int8()))

    tenure_bucket = _tenure_bucket_id(tenure)

    short_tenure = _flag(pc.less(tenure, 6))
    long_tenure = _flag(pc.greater(tenure, 24))
    is_month_to_month = _flag(pc.equal(contract, "month-to-month"))
    has_fiber = _flag(pc.equal(internet_service, "fiber optic"))
    no_internet = _flag(pc.equal(internet_service, "no"))
    high_monthly_charges = _flag(pc.greater(monthly_charges, 80))

    return pa.table(
        {
            "churn": churn01,
            "customer_id": customer_id,

            "senior_citizen": senior_citizen,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,

            "gender_id": gender01,
            "partner_id": partner01,
            "dependents_id": dependents01,
            "phone_service_id": phone_service01,
            "multiple_lines_id": multiple_lines01,
            "internet_service_id": internet_service01,
            "online_security_id": online_security_id,
            "online_backup_id": online_backup_id,
            "device_protection_id": device_protection_id,
            "tech_support_id": tech_support_id,
            "streaming_tv_id": streaming_tv_id,
            "streaming_movies_id": streaming_movies_id,
            "contract_term_months": contract_months,
            "paperless_billing_id": paperless01,
            "payment_method_id": payment_id,

            "num_services": num_services,
            "num_addon_services": num_addon_services,

            "security_bundle": security_bundle,
            "tech_bundle": tech_bundle,

            "charges_per_month": charges_per_month,
            "charges_ratio": charges_ratio,
            "addon_service_ratio": addon_service_ratio,

            "tenure_bucket_id": tenure_bucket,
            "short_tenure": short_tenure,
            "long_tenure": long_tenure,
            "is_month_to_month": is_month_to_month,
            "has_fiber": has_fiber,
            "no_internet": no_internet,
            "high_monthly_charges": high_monthly_charges,
        }
    )
