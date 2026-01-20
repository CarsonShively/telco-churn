import pandas as pd
import dagster as dg
from telco_churn.batch.scored import build_scored_df

@dg.asset(name="score", required_resource_keys={"hf_model", "batch_ctx"})
def score(context: dg.AssetExecutionContext, df_features: pd.DataFrame) -> pd.DataFrame:
    hf_model = context.resources.hf_model
    batch_ctx = context.resources.batch_ctx
    ctx = batch_ctx.get()
    bundle = hf_model.get_model_bundle()
        
    proba = bundle.model.predict_proba(df_features)[:, 1]
    
    return build_scored_df(
        X=df_features,
        proba=proba,
        batch_id=ctx.batch_id,
        threshold=bundle.threshold,
    )