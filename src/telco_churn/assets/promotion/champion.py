import dagster as dg
from typing import Optional, Any
import pandas as pd

@dg.asset(name="champion", required_resource_keys={"hf_model"})
def champion(context: dg.AssetExecutionContext) -> Optional[dict[str, Any]]:
    hf_model = context.resources.hf_model
    champion_ptr = hf_model.model_json(path_in_repo="champion.json")

    metrics_path = f'{champion_ptr["path_in_repo"]}/metrics.json'
    metrics = hf_model.model_json(path_in_repo=metrics_path)

    context.add_output_metadata({
        "champion_ptr_path": str(champion_ptr.get("path_in_repo")),
        "metrics_path": metrics_path,
        "keys": list(metrics.keys()),
        "preview": dg.MetadataValue.md(
            pd.DataFrame(list(metrics.items())[:10], columns=["key", "value"]).to_markdown(index=False)
        ),
    })

    return metrics
