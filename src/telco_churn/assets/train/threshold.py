import dagster as dg
from telco_churn.modeling.threshold import tune_threshold
from telco_churn.config import FLAG_RATE
from telco_churn.modeling.types import TTSCV, FitOut

@dg.asset(name="best_threshold")
def threshold(context: dg.AssetExecutionContext, data_splits: TTSCV, fit_pipeline: FitOut) -> float:
    holdout_scores = fit_pipeline.artifact.predict_proba(data_splits.X_holdout)[:, 1]
    thr = float(tune_threshold(holdout_scores, flag_rate=FLAG_RATE))

    context.add_output_metadata({
        "flag_rate": float(FLAG_RATE),
        "holdout_rows": int(data_splits.X_holdout.shape[0]),
        "holdout_cols": int(data_splits.X_holdout.shape[1]),
        "threshold": thr,
    })

    return thr