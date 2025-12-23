import argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

from telco_churn.modeling.run_id import make_run_id
from telco_churn.modeling.metrics.report import project_metric_report
from telco_churn.io.hf import download_from_hf, upload_bundle
from telco_churn.modeling.dataframe.train_to_df import parquet_to_df
from telco_churn.modeling.assemble_bundle.assemble_bundle import write_bundle
from telco_churn.modeling.optuna import tune_optuna_cv
from telco_churn.modeling.train_pipeline_ops import fit_best, evaluate
from telco_churn.modeling.trainers.make_trainer import make_trainer, available_trainers
from telco_churn.modeling.assemble_bundle.assemble_artifact import ModelArtifact

from telco_churn.config import (
    REPO_ID,
    REVISION,
    TRAIN_HF_PATH,
    TARGET_COL,
    PRIMARY_METRIC,
    METRIC_DIRECTION,
    DEFAULT_THRESHOLD,
)


HOLDOUT_SIZE = 0.2
SEED = 42
CV_SPLITS = 5
N_TRIALS = 250

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--modeltype", required=True, choices=available_trainers())
    p.add_argument(
        "--upload",
        action="store_true",
        help="Upload the run bundle to Hugging Face after training",
    )
    return p.parse_args()


def main(*, modeltype: str, upload: bool = False) -> None:
    run_id = make_run_id()

    metrics = project_metric_report()

    if PRIMARY_METRIC not in metrics:
        raise ValueError(
            f"Unknown primary_metric {PRIMARY_METRIC!r}. Options: {sorted(metrics.keys())}"
        )

    primary_metric_fn = metrics[PRIMARY_METRIC]

    trainer = make_trainer(modeltype, seed=SEED)

    local_path = download_from_hf(
        repo_id=REPO_ID,
        filename=TRAIN_HF_PATH,
        revision=REVISION,
    )
    df = parquet_to_df(local_path)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}"
        )

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    if not (0.0 < HOLDOUT_SIZE < 1.0):
        raise ValueError(f"HOLDOUT_SIZE must be in (0, 1), got {HOLDOUT_SIZE}")

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=HOLDOUT_SIZE,
        random_state=SEED,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    best_params, cv_summary = tune_optuna_cv(
        build_pipeline=trainer.build_pipeline,
        suggest_params=trainer.suggest_params,
        X=X_train,
        y=y_train,
        cv=cv,
        primary_metric=primary_metric_fn,
        metrics=metrics,
        direction=METRIC_DIRECTION,
        n_trials=N_TRIALS,
        seed=SEED,
    )

    artifact = fit_best(
        build_pipeline=trainer.build_pipeline,
        X=X_train,
        y=y_train,
        best_params=best_params,
    )

    holdout_metrics = evaluate(
        artifact,
        X_holdout,
        y_holdout,
        metrics=metrics,
        threshold=DEFAULT_THRESHOLD,
    )

    artifact_obj = ModelArtifact(
        run_id=run_id,
        model_type=modeltype,
        model=artifact,
        threshold=DEFAULT_THRESHOLD,
    )

    bundle_dir = Path("artifacts/runs") / run_id

    bundle_dir = write_bundle(
        bundle_dir=bundle_dir,
        artifact_obj=artifact_obj,
        best_params=best_params,
        cv_summary=cv_summary,
        holdout_metrics=holdout_metrics,
        primary_metric=PRIMARY_METRIC,
        direction=METRIC_DIRECTION,
    )

    required = ["model.joblib", "metrics.json", "metadata.json"]
    missing = [name for name in required if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Bundle missing required files: {missing} in {bundle_dir}"
        )

    if upload:
        upload_bundle(
            bundle_dir=bundle_dir,
            repo_id=REPO_ID,
            run_id=run_id,
            revision=REVISION,
        )

if __name__ == "__main__":
    args = parse_args()
    main(modeltype=args.modeltype, upload=args.upload)
