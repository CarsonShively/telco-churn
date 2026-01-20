import pandas as pd
import dagster as dg
from telco_churn.batch.action import build_actions_df

@dg.asset(name="action", required_resource_keys={"hf_model"})
def action(context: dg.AssetExecutionContext, df_features: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    hf_model = context.resources.hf_model
    bundle = hf_model.get_model_bundle()

    return build_actions_df(
        scored=score,
        X=df_features,
        model=bundle.model,
        names=bundle.feature_names,
    )