import os
import dagster as dg

@dg.asset(name="train_data", required_resource_keys={"hf_data"})
def train_data(context: dg.AssetExecutionContext) -> str:
    hf_data = context.resources.hf_data
    local_path = hf_data.download_data("data/gold/train.parquet")

    local_path = str(local_path)

    context.add_output_metadata({
        "path": dg.MetadataValue.path(local_path),
        "bytes": os.path.getsize(local_path),
    })

    return local_path
