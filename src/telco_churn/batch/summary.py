"""
Summarize batch report.
"""

from datetime import datetime, timezone
import pandas as pd

def build_batch_summary_core(
    *,
    batch_id: str,
    model_version: str,
    threshold: float,
    scored: pd.DataFrame,
    actions: pd.DataFrame,
    top_k: int = 3,
) -> dict:
    total_scored = int(len(scored))
    flagged_count = int(scored["decision"].sum())
    flag_rate = float(flagged_count / total_scored) if total_scored else 0.0

    s = scored["probability"].astype(float)
    score_mean = float(s.mean()) if total_scored else 0.0
    score_median = float(s.median()) if total_scored else 0.0
    score_p95 = float(s.quantile(0.95)) if total_scored else 0.0

    bucket_counts = scored["risk_bucket"].astype("string").value_counts().to_dict()
    risk_bucket_counts = {
        "low": int(bucket_counts.get("low", 0)),
        "medium": int(bucket_counts.get("medium", 0)),
        "high": int(bucket_counts.get("high", 0)),
    }

    actions_count = int(len(actions))

    if actions_count and "reason_code" in actions.columns:
        rc = actions["reason_code"].astype("string").value_counts()
        top_reason_codes = [
            {"reason_code": str(code), "count": int(cnt), "share": float(cnt / actions_count)}
            for code, cnt in rc.head(top_k).items()
        ]
    else:
        top_reason_codes = []

    return {
        "batch_id": batch_id,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "threshold": float(threshold),

        "total_scored": total_scored,
        "flagged_count": flagged_count,
        "flag_rate": float(flag_rate),

        "score_mean": float(score_mean),
        "score_median": float(score_median),
        "score_p95": float(score_p95),

        "risk_bucket_counts": risk_bucket_counts,

        "actions_count": actions_count,
        "top_reason_codes": top_reason_codes,
    }


