import dagster as dg

@dg.asset(name="train_data", required_resource_keys={"hf_data"})
def train_data(context: dg.AssetExecutionContext) -> str:
    hf_data = context.resources.hf_data
    return hf_data.download_data("data/gold/train.parquet")