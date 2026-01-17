import pandas as pd

from telco_churn.explainability.decision_codes import DECISION_CODES
from telco_churn.explainability.action_map import DECISION_ACTIONS
from telco_churn.explainability.explain import top_feature_name, build_explainer

def build_actions_df(
    *,
    scored: pd.DataFrame,
    X: pd.DataFrame,
    model,
    names,
) -> pd.DataFrame | None:
    actions = scored.loc[scored["decision"].eq(1)].copy()
    
    if actions.empty:
        return None
    
    flagged_ids = actions["customer_id"].unique()
    X_flagged = X.loc[X["customer_id"].isin(flagged_ids)].copy()
    
    explainer = build_explainer(pipe=model)

    top_feature_list: list[str] = []
    for _, r in X_flagged.iterrows():
        X_row = pd.DataFrame([r.to_dict()])
        top_feature_list.append(top_feature_name(pipe=model, feature_names=names, explainer=explainer, X_row=X_row))

    X_flagged["top_feature"] = top_feature_list
    
    actions = actions.merge(
        X_flagged[["customer_id", "top_feature"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    
    actions["reason_code"] = actions["top_feature"].map(DECISION_CODES).fillna("OTHER")
    actions["recommended_action"] = actions["reason_code"].map(DECISION_ACTIONS).fillna(
        DECISION_ACTIONS["OTHER"]
    )
    
    actions["action_summary"] = actions.apply(
        lambda r: f'{r["reason_code"]}: {r["recommended_action"]}',
        axis=1,
    )

    actions_cols = [
        "priority_rank",
        "batch_id",
        "customer_id",
        "probability",
        "risk_bucket",
        "reason_code",
        "recommended_action",
        "action_summary",
    ]
    
    return actions.sort_values("priority_rank").reset_index(drop=True)[actions_cols]