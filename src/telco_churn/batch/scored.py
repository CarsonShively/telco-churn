"""
Scored batch dataframe.
"""

import pandas as pd

def build_scored_df(
    *,
    X: pd.DataFrame,
    proba: pd.Series,
    batch_id: str,
    threshold: float,
) -> pd.DataFrame:
    scored = X.copy()
    
    scored["batch_id"] = batch_id

    scored["probability"] = proba

    scored["threshold"] = float(threshold)
    
    scored["decision"] = (scored["probability"] >= scored["threshold"]).astype("int8")

    scored["risk_bucket"] = pd.cut(
        scored["probability"],
        bins=[-1, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"],
    )

    scored["priority_rank"] = scored["probability"].rank(
        ascending=False,
        method="first",
    ).astype("int32")

    cols = [
        "batch_id",
        "customer_id",
        "probability",
        "decision",
        "threshold",
        "risk_bucket",
        "priority_rank",
    ]
    return scored[cols]