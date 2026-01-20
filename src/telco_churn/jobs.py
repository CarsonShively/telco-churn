import dagster as dg

data_job = dg.define_asset_job(
    "data_job",
    selection=dg.AssetSelection.keys("materialise_gold").upstream(),
    executor_def=dg.in_process_executor,
)

batch_job = dg.define_asset_job(
    "batch_job",
    selection=dg.AssetSelection.keys("upload_report").upstream(),
    executor_def=dg.in_process_executor,
)

train_job = dg.define_asset_job(
    "train_job",
    selection=dg.AssetSelection.keys("upload_bundle").upstream(),
    executor_def=dg.in_process_executor,
)