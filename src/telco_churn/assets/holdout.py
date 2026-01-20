import dagster as dg

from telco_churn.modeling.evaluate import evaluate
from telco_churn.modeling.metrics.report import project_metric_report
from telco_churn.modeling.types import TTSCV, FitOut

@dg.asset(name="holdout")
def holdout(tts_cv: TTSCV, fit: FitOut, threshold: float) -> dict[str, float]:
    metrics = project_metric_report()

    out = evaluate(
        fit.artifact,
        tts_cv.X_holdout,
        tts_cv.y_holdout,
        metrics=metrics,
        threshold=threshold,
    )

    return {k: float(v) for k, v in out.items()}
