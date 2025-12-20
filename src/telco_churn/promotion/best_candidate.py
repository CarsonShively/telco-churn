from typing import Any, Dict, List, Optional, Tuple

def primary_metric_name(m: Dict[str, Any]) -> str:
    pm = m.get("primary_metric")
    if not pm:
        raise ValueError("metrics.json missing required field: 'primary_metric'")
    return pm


def passes_selection_gates(
    m: Dict[str, Any], *, min_recall: float = 0.20, min_precision: float = 0.10
) -> bool:
    pm = primary_metric_name(m)

    holdout = m.get("holdout", {})
    primary_val = holdout.get(pm)
    rec = holdout.get("recall")
    prec = holdout.get("precision")

    return (
        primary_val is not None
        and rec is not None and rec >= min_recall
        and prec is not None and prec >= min_precision
    )


def cv_primary_mean(m: Dict[str, Any]) -> Optional[float]:
    pm = primary_metric_name(m)
    return m.get("cv", {}).get("metrics", {}).get(pm, {}).get("mean")


def cv_primary_std(m: Dict[str, Any]) -> Optional[float]:
    pm = primary_metric_name(m)
    return m.get("cv", {}).get("metrics", {}).get(pm, {}).get("std")


def holdout_primary_value(m: Dict[str, Any]) -> Optional[float]:
    pm = primary_metric_name(m)
    return m.get("holdout", {}).get(pm)


def pick_best_contender(rows) -> Any:
    candidates: list[tuple[float, float, str, Any]] = []

    for r in rows:
        if getattr(r, "error", None):
            continue

        m = r.metrics
        pm = m.get("primary_metric")
        if not pm:
            continue

        if not passes_selection_gates(m):
            continue

        mean = cv_primary_mean(m)
        std = cv_primary_std(m)
        if mean is None:
            continue

        std_key = -(std if std is not None else float("inf"))

        run_id = getattr(r, "run_id", "")
        candidates.append((mean, std_key, run_id, r))

    if not candidates:
        raise ValueError("No candidates passed gates and had required CV primary metric")

    candidates.sort(reverse=True)
    return candidates[0][3]