import dagster as dg

from telco_churn.modeling.evaluate import evaluate
from telco_churn.modeling.metrics.report import project_metric_report
from telco_churn.modeling.types import TTSCV, FitOut

@dg.asset(name="holdout_evaluation")
def holdout_evaluation(data_splits: TTSCV, fit_pipeline: FitOut, best_threshold: float) -> dict[str, float]:
    metrics = project_metric_report()

    out = evaluate(
        fit_pipeline.artifact,
        data_splits.X_holdout,
        data_splits.y_holdout,
        metrics=metrics,
        threshold=best_threshold,
    )

    return {k: float(v) for k, v in out.items()}
