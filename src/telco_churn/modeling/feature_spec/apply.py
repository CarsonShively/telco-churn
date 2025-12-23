"""Pure feature-spec application: add missing columns and enforce pandas dtypes."""

from __future__ import annotations

from typing import Any, Dict
import pandas as pd

_DTYPE_MAP = {
    "string": "string",
    "int": "Int64",
    "float": "Float64",
    "bool": "boolean",
    "category": "category",
}

def feature_spec(
    df: pd.DataFrame,
    spec: Dict[str, Any],
    *,
    add_missing_columns: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    features = spec.get("features", [])
    for f in features:
        name = f["name"]
        dtype = f.get("dtype")

        if add_missing_columns and name not in out.columns:
            out[name] = pd.NA

        if name not in out.columns:
            continue

        if dtype:
            pandas_dtype = _DTYPE_MAP.get(dtype)
            if pandas_dtype is None:
                raise ValueError(f"Unknown dtype {dtype!r} for feature {name!r}")

            out[name] = out[name].astype(pandas_dtype)

    return out
