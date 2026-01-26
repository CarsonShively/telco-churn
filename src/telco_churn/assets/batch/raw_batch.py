import os
import pandas as pd
import dagster as dg

@dg.asset(name="raw_batch", required_resource_keys={"hf_data"})
def raw_batch(context: dg.AssetExecutionContext) -> str:
    """Incoming batch data."""
    hf_data = context.resources.hf_data
    local_path = hf_data.download_data("data/bronze/demo.parquet")

    file_bytes = os.path.getsize(local_path)

    df_head = pd.read_parquet(local_path).head(5)

    context.add_output_metadata({
        "path": dg.MetadataValue.path(local_path),
        "bytes": file_bytes,
        "rows_previewed": len(df_head),
        "columns": len(df_head.columns),
        "preview": dg.MetadataValue.md(df_head.to_markdown(index=False)),
    })

    return local_path