import dagster as dg

@dg.asset(name="raw_batch", required_resource_keys={"hf_data"})
def raw_batch(context: dg.AssetExecutionContext) -> str:
    hf_data = context.resources.hf_data
    return hf_data.download_data("data/bronze/demo.parquet")