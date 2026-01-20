import pandas as pd
import dagster as dg
from sklearn.model_selection import StratifiedKFold, train_test_split
from telco_churn.modeling.config import TARGET_COL, HOLDOUT_SIZE, SEED, CV_SPLITS
from telco_churn.modeling.types import TTSCV

@dg.asset(name="tts_cv")
def tts_cv(model_data: str) -> TTSCV:
    df = pd.read_parquet(model_data)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y,
        test_size=HOLDOUT_SIZE,
        random_state=SEED,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)

    return TTSCV(X_train, X_holdout, y_train, y_holdout, cv)