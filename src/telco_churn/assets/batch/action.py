import pandas as pd
import dagster as dg
from telco_churn.batch.action import build_actions_df

@dg.asset(name="batch_action_df", required_resource_keys={"hf_model"})
def batch_action_df(context: dg.AssetExecutionContext, batch_features_df: pd.DataFrame, batch_scored_df: pd.DataFrame) -> pd.DataFrame:
    hf_model = context.resources.hf_model
    bundle = hf_model.get_model_bundle()

    return build_actions_df(
        scored=batch_scored_df,
        X=batch_features_df,
        model=bundle.model,
        names=bundle.feature_names,
    )