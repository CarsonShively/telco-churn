from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd


def coerce_df_to_feature_spec(
    df: pd.DataFrame,
    spec: Dict[str, Any],
    *,
    add_missing_columns: bool = True,
    fill_defaults: bool = True,
    enforce_required: bool = True,
) -> pd.DataFrame:
    """
    Coerce/align a DataFrame to a feature spec:
      - optionally adds missing columns
      - optionally fills defaults
      - coerces dtypes (string/int/float)
      - optionally enforces required columns exist (and non-null if default is null)
      - returns columns ordered as spec["features"]

    Spec format expected:
      {"features":[{"name":..., "dtype": "string|int|float", "required": bool, "default": ...}, ...]}
    """
    if "features" not in spec or not isinstance(spec["features"], list):
        raise ValueError("spec must contain a list at spec['features'].")

    out = df.copy()

    def _is_nullish(x: Any) -> bool:
        return x is None or (isinstance(x, float) and pd.isna(x))

    def _cast_series(s: pd.Series, dtype: str) -> pd.Series:
        dtype = dtype.lower().strip()
        if dtype == "string":
            return s.astype("string")
        if dtype == "int":
            return pd.to_numeric(s, errors="coerce").astype("Int64")
        if dtype == "float":
            return pd.to_numeric(s, errors="coerce").astype("Float64")
        raise ValueError(f"Unsupported dtype in spec: {dtype!r}")

    required_missing = []
    required_null_violations = []

    for f in spec["features"]:
        name = f["name"]
        required = bool(f.get("required", False))
        default = f.get("default", None)

        if name not in out.columns:
            if add_missing_columns:
                out[name] = pd.NA
            elif required:
                required_missing.append(name)
                continue
            else:
                continue

        if fill_defaults and not _is_nullish(default):
            out[name] = out[name].fillna(default)

        out[name] = _cast_series(out[name], f["dtype"])

        if enforce_required and required:
            if _is_nullish(default) and out[name].isna().any():
                required_null_violations.append(name)

    if enforce_required:
        if required_missing:
            raise ValueError(f"Missing required columns: {required_missing}")
        if required_null_violations:
            raise ValueError(
                "Required columns contain nulls (and default is null): "
                f"{required_null_violations}"
            )

    ordered_cols = [f["name"] for f in spec["features"] if f["name"] in out.columns]
    return out[ordered_cols]
