"""
Jobs:
    ETL = raw data to train ready data.
    Train = train model artifact.
    Promotion = determine best current model artifact.
    Batch = score incoming batch data. 
"""

import dagster as dg

etl = dg.define_asset_job(
    "etl",
    selection=dg.AssetSelection.keys("upload_train_table").upstream(),
    executor_def=dg.in_process_executor,
)

train = dg.define_asset_job(
    "train",
    selection=dg.AssetSelection.keys("upload_bundle").upstream(),
    executor_def=dg.in_process_executor,
)

promotion = dg.define_asset_job(
    "promotion",
    selection=dg.AssetSelection.keys("execute_promotion_decision").upstream(),
    executor_def=dg.in_process_executor,
)

batch = dg.define_asset_job(
    "batch",
    selection=dg.AssetSelection.keys("upload_batch_report").upstream(),
    executor_def=dg.in_process_executor,
)