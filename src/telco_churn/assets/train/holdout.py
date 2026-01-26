import dagster as dg

from telco_churn.modeling.evaluate import evaluate
from telco_churn.modeling.metrics.report import project_metric_report
from telco_churn.modeling.types import TTSCV, FitOut

@dg.asset(name="holdout_evaluation")
def holdout_evaluation(
    context: dg.AssetExecutionContext,
    data_splits: TTSCV,
    fit_pipeline: FitOut,
    best_threshold: float,
) -> dict[str, float]:
    """Evaluate on holdout split and get metrics."""
    metrics = project_metric_report()

    out = evaluate(
        fit_pipeline.artifact,
        data_splits.X_holdout,
        data_splits.y_holdout,
        metrics=metrics,
        threshold=best_threshold,
    )

    out_f = {k: float(v) for k, v in out.items()}

    context.add_output_metadata({
        "threshold": float(best_threshold),
        "holdout_rows": int(data_splits.X_holdout.shape[0]),
        "holdout_cols": int(data_splits.X_holdout.shape[1]),
        "metrics": out_f,
        "preview": dg.MetadataValue.md(
            "\n".join([f"- `{k}`: {v}" for k, v in out_f.items()])
        ),
    })

    return out_f
