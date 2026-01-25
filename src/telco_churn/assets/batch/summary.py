import pandas as pd
import dagster as dg
from telco_churn.batch.summary import build_batch_summary_core

@dg.asset(name="batch_summary", required_resource_keys={"hf_model", "batch_ctx"})
def batch_summary(
    context: dg.AssetExecutionContext,
    batch_scored_df: pd.DataFrame,
    batch_action_df: pd.DataFrame,
) -> dict:
    hf_model = context.resources.hf_model
    batch_ctx = context.resources.batch_ctx
    ctx = batch_ctx.get()
    bundle = hf_model.get_model_bundle()

    summary = build_batch_summary_core(
        batch_id=ctx.batch_id,
        model_version=bundle.model_version,
        threshold=bundle.threshold,
        scored=batch_scored_df,
        actions=batch_action_df,
        top_k=3,
    )

    context.add_output_metadata({
        "batch_id": summary["batch_id"],
        "scored_at_utc": summary["scored_at_utc"],
        "model_version": summary["model_version"],
        "threshold": summary["threshold"],

        "total_scored": summary["total_scored"],
        "flagged_count": summary["flagged_count"],
        "flag_rate": summary["flag_rate"],

        "score_mean": summary["score_mean"],
        "score_median": summary["score_median"],
        "score_p95": summary["score_p95"],

        "risk_bucket_counts": summary["risk_bucket_counts"],
        "actions_count": summary["actions_count"],

        "preview": dg.MetadataValue.md(
            pd.DataFrame(summary.items(), columns=["key", "value"]).to_markdown(index=False)
        ),
    })

    return summary
