import dagster as dg
from telco_churn.modeling.threshold import tune_threshold
from telco_churn.config import FLAG_RATE
from telco_churn.modeling.types import TTSCV, FitOut

@dg.asset(name="threshold")
def threshold(tts_cv: TTSCV, fit: FitOut) -> float:
    holdout_scores = fit.artifact.predict_proba(tts_cv.X_holdout)[:, 1]
    return float(tune_threshold(holdout_scores, flag_rate=FLAG_RATE))
    