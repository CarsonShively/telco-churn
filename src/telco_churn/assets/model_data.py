import dagster as dg

@dg.asset(name="model_data", required_resource_keys={"hf_data"})
def ingest_model_data(context: dg.AssetExecutionContext) -> str:
    hf_data = context.resources.hf_data
    return hf_data.download_data("data/gold/train.parquet")