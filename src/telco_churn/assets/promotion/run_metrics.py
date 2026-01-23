import dagster as dg
from telco_churn.io.hf_run_metrics import RunRow

@dg.asset(name="run_metrics", required_resource_keys={"hf_model"})
def run_metrics(context: dg.AssetExecutionContext) -> list[RunRow]:
    hf_model = context.resources.hf_model
    return hf_model.run_metrics()