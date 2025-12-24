"""Select the best run by ranking candidates on primary-metric CV mean (then lower CV std, then holdout score, then run_id)."""

from __future__ import annotations

from typing import Any, Optional
import math
from telco_churn.config import CURRENT_ARTIFACT_VERSION


def _f(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    return None

def _i(x: Any) -> Optional[int]:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return None

def primary_metric_name(m: dict[str, Any]) -> str:
    pm = m.get("primary_metric")
    if not isinstance(pm, str) or not pm:
        raise ValueError("metrics.json missing required field: 'primary_metric'")
    return pm

def artifact_version(m: dict[str, Any]) -> Optional[int]:
    return _i(m.get("artifact_version"))

def get_best_contender(rows: list[Any]) -> Any:
    best: Any = None
    best_key: Optional[tuple] = None

    for r in rows:
        if getattr(r, "error", None):
            continue

        m = getattr(r, "metrics", None) or {}
        
        ver = artifact_version(m)
        if ver != CURRENT_ARTIFACT_VERSION:
            continue
        
        try:
            pm = primary_metric_name(m)
        except ValueError:
            continue

        hold = m.get("holdout", {}) or {}
        cvm = ((m.get("cv", {}) or {}).get("metrics", {}) or {}).get(pm, {}) or {}

        cv_mean = _f(cvm.get("mean"))
        hold_pm = _f(hold.get(pm))
        if cv_mean is None or hold_pm is None:
            continue

        cv_std = _f(cvm.get("std"))
        run_id = getattr(r, "run_id", "") or ""

        key = (
            cv_mean,
            -(cv_std if cv_std is not None else float("inf")),
            hold_pm,
            run_id,
        )

        if best_key is None or key > best_key:
            best_key = key
            best = r

    if best is None:
        raise ValueError("No candidates passed gates")

    return best