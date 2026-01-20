import pandas as pd
import dagster as dg
from telco_churn.batch.summary import build_batch_summary_core

@dg.asset(name="summary", required_resource_keys={"hf_model", "batch_ctx"})
def summary(context: dg.AssetExecutionContext, score: pd.DataFrame, action: pd.DataFrame) -> dict:
    hf_model = context.resources.hf_model
    batch_ctx = context.resources.batch_ctx
    ctx = batch_ctx.get()
    bundle = hf_model.get_model_bundle()

    return build_batch_summary_core(
        batch_id=ctx.batch_id,
        model_version=bundle.model_version,
        threshold=bundle.threshold,
        scored=score,
        actions=action,
        top_k=3,
    )