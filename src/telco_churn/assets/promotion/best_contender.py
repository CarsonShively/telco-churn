import dagster as dg
import pandas as pd
from telco_churn.promotion.best_candidate import get_best_contender
from telco_churn.promotion.type import RunRow

@dg.asset(name="best_contender")
def best_contender(context: dg.AssetExecutionContext, run_metrics: list[RunRow]) -> RunRow:
    """Select best contender from model runs archive."""
    best = get_best_contender(run_metrics)

    context.add_output_metadata({
        "run_id": best.run_id,
        "model_type": best.model_type,
        "metrics_path": best.metrics_path,
        "error": best.error,
        "metrics_keys": list((best.metrics or {}).keys()),
        "preview": dg.MetadataValue.md(
            pd.DataFrame(list((best.metrics or {}).items())[:10], columns=["key", "value"])
            .to_markdown(index=False)
        ),
    })

    return best
