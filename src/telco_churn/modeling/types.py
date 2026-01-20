import pandas as pd
from enum import Enum
import dagster as dg
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict, Any
from sklearn.model_selection import StratifiedKFold

@dataclass(frozen=True)
class TTSCV:
    X_train: pd.DataFrame
    X_holdout: pd.DataFrame
    y_train: pd.Series
    y_holdout: pd.Series
    cv: StratifiedKFold

@dataclass(frozen=True)
class TuningResult:
    best_params: dict[str, Any]
    cv_summary: dict[str, Any]
    
class ModelType(str, Enum):
    lr = "lr"
    lgbm = "lgbm"
    xgb = "xgb"

class TrainConfig(dg.Config):
    model_type: ModelType = ModelType.lr
    n_trials: int = 10
    upload: bool = False
    
@dataclass(frozen=True)
class FitOut:
    artifact: Any
    feature_names: list[str]
    
@dataclass(frozen=True)
class BundleOut:
    run_id: str
    bundle_dir: Path