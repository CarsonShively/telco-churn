import pandas as pd
import dagster as dg
from telco_churn.batch.summary import build_batch_summary_core

@dg.asset(name="batch_summary", required_resource_keys={"hf_model", "batch_ctx"})
def batch_summary(context: dg.AssetExecutionContext, batch_scored_df: pd.DataFrame, batch_action_df: pd.DataFrame) -> dict:
    hf_model = context.resources.hf_model
    batch_ctx = context.resources.batch_ctx
    ctx = batch_ctx.get()
    bundle = hf_model.get_model_bundle()

    return build_batch_summary_core(
        batch_id=ctx.batch_id,
        model_version=bundle.model_version,
        threshold=bundle.threshold,
        scored=batch_scored_df,
        actions=batch_action_df,
        top_k=3,
    )