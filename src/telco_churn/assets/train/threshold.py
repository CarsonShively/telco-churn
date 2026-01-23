import dagster as dg
from telco_churn.modeling.threshold import tune_threshold
from telco_churn.config import FLAG_RATE
from telco_churn.modeling.types import TTSCV, FitOut

@dg.asset(name="best_threshold")
def threshold(data_splits: TTSCV, fit_pipeline: FitOut) -> float:
    holdout_scores = fit_pipeline.artifact.predict_proba(data_splits.X_holdout)[:, 1]
    return float(tune_threshold(holdout_scores, flag_rate=FLAG_RATE))
    