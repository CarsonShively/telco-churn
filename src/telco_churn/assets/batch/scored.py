import pandas as pd
import dagster as dg
from telco_churn.batch.scored import build_scored_df

@dg.asset(name="batch_scored_df", required_resource_keys={"hf_model", "batch_ctx"})
def batch_scored_df(context: dg.AssetExecutionContext, batch_features_df: pd.DataFrame) -> pd.DataFrame:
    hf_model = context.resources.hf_model
    batch_ctx = context.resources.batch_ctx
    ctx = batch_ctx.get()
    bundle = hf_model.get_model_bundle()
        
    proba = bundle.model.predict_proba(batch_features_df)[:, 1]
    
    return build_scored_df(
        X=batch_features_df,
        proba=proba,
        batch_id=ctx.batch_id,
        threshold=bundle.threshold,
    )