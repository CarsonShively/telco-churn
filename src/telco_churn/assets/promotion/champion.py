import dagster as dg
from typing import Optional, Any

@dg.asset(name="champion", required_resource_keys={"hf_model"})
def champion(context: dg.AssetExecutionContext) -> Optional[dict[str, Any]]:
    hf_model = context.resources.hf_model
    champion_ptr = hf_model.model_json(path_in_repo="champion.json")
    return hf_model.model_json(path_in_repo=f'{champion_ptr["path_in_repo"]}/metrics.json')