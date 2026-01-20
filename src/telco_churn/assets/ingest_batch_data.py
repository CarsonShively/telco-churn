import dagster as dg

@dg.asset(name="ingest_batch_data", required_resource_keys={"hf_data"})
def ingest_batch_data(context: dg.AssetExecutionContext) -> str:
    hf_data = context.resources.hf_data
    return hf_data.download_data("data/bronze/demo.parquet")