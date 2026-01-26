import pandas as pd
import dagster as dg
from sklearn.model_selection import StratifiedKFold, train_test_split
from telco_churn.modeling.config import TARGET_COL, HOLDOUT_SIZE, SEED, CV_SPLITS
from telco_churn.modeling.types import TTSCV

@dg.asset(name="data_splits")
def data_splits(context: dg.AssetExecutionContext, train_data: str) -> TTSCV:
    """Establish data splits: TTS and CV."""
    df = pd.read_parquet(train_data)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y,
        test_size=HOLDOUT_SIZE,
        random_state=SEED,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)

    context.add_output_metadata({
        "train_path": dg.MetadataValue.path(str(train_data)),
        "target_col": TARGET_COL,
        "holdout_size": float(HOLDOUT_SIZE),
        "seed": int(SEED),
        "cv_splits": int(CV_SPLITS),

        "X_train_rows": int(X_train.shape[0]),
        "X_train_cols": int(X_train.shape[1]),
        "X_holdout_rows": int(X_holdout.shape[0]),
        "X_holdout_cols": int(X_holdout.shape[1]),

        "y_train_counts": y_train.value_counts(dropna=False).to_dict(),
        "y_holdout_counts": y_holdout.value_counts(dropna=False).to_dict(),

        "X_train_preview": dg.MetadataValue.md(X_train.head(5).to_markdown(index=False)),
    })

    return TTSCV(X_train, X_holdout, y_train, y_holdout, cv)
