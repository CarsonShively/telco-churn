import dagster as dg
from telco_churn.assets.ingest_train_data import ingest_train_data
from telco_churn.resources.duckdb import DuckDBResource
from telco_churn.assets.bronze_train import bronze_train_table
from telco_churn.assets.labels import labels_table
from telco_churn.assets.train_base import train_base_table
from telco_churn.assets.train_features import train_features_table
from telco_churn.assets.train_join_labels import train_join_table
from telco_churn.assets.materialise_gold import materiailise_gold

from telco_churn.assets.ingest_batch_data import ingest_batch_data
from telco_churn.assets.bronze_batch import bronze_batch_table
from telco_churn.assets.batch_base import batch_base_table
from telco_churn.assets.batch_features import batch_features_table
from telco_churn.assets.df_features import df_features
from telco_churn.assets.score import score
from telco_churn.assets.action import action
from telco_churn.assets.summary import summary
from telco_churn.assets.report import report
from telco_churn.assets.upload_report import upload_report

from telco_churn.assets.model_data import ingest_model_data
from telco_churn.assets.params import best_params
from telco_churn.assets.bundle import bundle
from telco_churn.assets.fit_artifact import fit_artifact
from telco_churn.assets.holdout import holdout
from telco_churn.assets.threshold import threshold
from telco_churn.assets.tts_cv import tts_cv
from telco_churn.assets.upload_bundle import upload_bundle

from telco_churn.jobs import data_job, batch_job, train_job
from telco_churn.resources.data import HFDataResource
from telco_churn.resources.model import HFModelResource
from telco_churn.resources.batch import BatchContextResource
from telco_churn.config import REPO_ID, REVISION


defs = dg.Definitions(
    assets=[
        ingest_train_data,
        bronze_train_table,
        labels_table,
        train_base_table,
        train_features_table,
        train_join_table,
        materiailise_gold,
        ingest_batch_data,
        bronze_batch_table,
        batch_base_table,
        batch_features_table,
        df_features,
        score,
        action,
        summary,
        report,
        upload_report,
        ingest_model_data,
        tts_cv,
        best_params,
        threshold,
        fit_artifact,
        holdout,
        bundle,
        upload_bundle       
        
    ],
    jobs=[data_job, batch_job, train_job],
    resources={
        "hf_data": HFDataResource(repo_id=REPO_ID, revision=REVISION),
        "hf_model": HFModelResource(repo_id=REPO_ID, revision=REVISION),
        "batch_ctx": BatchContextResource(repo_root=".", reports_dirname="reports"),
        "db": DuckDBResource(path="data/telco.duckdb"),
        
    },
)