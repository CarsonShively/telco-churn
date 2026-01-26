import dagster as dg
import pandas as pd
from telco_churn.io.hf_run_metrics import RunRow

@dg.asset(name="run_metrics", required_resource_keys={"hf_model"})
def run_metrics(context: dg.AssetExecutionContext) -> list[RunRow]:
    """Retrive model run metrics from archive."""
    hf_model = context.resources.hf_model
    rows = hf_model.run_metrics()

    error_count = sum(1 for r in rows if r.error is not None)

    preview_df = pd.DataFrame([{
        "run_id": r.run_id,
        "model_type": r.model_type,
        "metrics_path": r.metrics_path,
        "error": r.error,
    } for r in rows[:5]])

    context.add_output_metadata({
        "count": len(rows),
        "error_count": int(error_count),
        "preview": dg.MetadataValue.md(preview_df.to_markdown(index=False)),
    })

    return rows
