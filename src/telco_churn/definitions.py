import dagster as dg

from telco_churn.assets.batch import BATCH_ASSET_MODULES
from telco_churn.assets.train import TRAIN_ASSET_MODULES
from telco_churn.assets.data import DATA_ASSET_MODULES
from telco_churn.assets.promotion import PROMOTION_ASSET_MODULES

from telco_churn.jobs import data, batch, train, promotion
from telco_churn.resources.duckdb import DuckDBResource
from telco_churn.resources.data import HFDataResource
from telco_churn.resources.model import HFModelResource
from telco_churn.resources.batch import BatchContextResource
from telco_churn.resources.train import TrainConfig
from telco_churn.config import REPO_ID, REVISION


all_assets = dg.load_assets_from_modules(
    TRAIN_ASSET_MODULES
    + BATCH_ASSET_MODULES
    + DATA_ASSET_MODULES
    + PROMOTION_ASSET_MODULES
)

defs = dg.Definitions(
    assets=all_assets,
    jobs=[data, batch, train, promotion],
    resources={
        "hf_data": HFDataResource(repo_id=REPO_ID, revision=REVISION),
        "hf_model": HFModelResource(repo_id=REPO_ID, revision=REVISION),
        "batch_ctx": BatchContextResource(repo_root=".", reports_dirname="reports"),
        "db": DuckDBResource(path="data/telco.duckdb"),
        "train_cfg": TrainConfig()
    },
)
