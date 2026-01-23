import dagster as dg

data = dg.define_asset_job(
    "data",
    selection=dg.AssetSelection.keys("upload_train_table").upstream(),
    executor_def=dg.in_process_executor,
)

batch = dg.define_asset_job(
    "batch",
    selection=dg.AssetSelection.keys("upload_batch_report").upstream(),
    executor_def=dg.in_process_executor,
)

train = dg.define_asset_job(
    "train",
    selection=dg.AssetSelection.keys("upload_bundle").upstream(),
    executor_def=dg.in_process_executor,
)

promotion = dg.define_asset_job(
    "promotion",
    selection=dg.AssetSelection.keys("promote").upstream(),
    executor_def=dg.in_process_executor,
)