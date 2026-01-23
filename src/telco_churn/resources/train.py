import dagster as dg
from enum import Enum

class ModelType(str, Enum):
    lr = "lr"
    lgb = "lgb"
    xgb = "xgb"

class TrainConfig(dg.ConfigurableResource):
    model_type: ModelType = ModelType.lr
    n_trials: int = 10
    upload: bool = False
