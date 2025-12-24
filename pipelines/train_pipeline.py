"""Train a churn model and write a run bundle.

Steps:
- Download gold train parquet from Hugging Face
- Train with Optuna CV, fit best model, evaluate holdout
- Write model.joblib, metrics.json, metadata.json
- Optionally upload the bundle to Hugging Face
"""

import argparse
import logging
from pathlib import Path
from time import perf_counter
import warnings

from sklearn.model_selection import StratifiedKFold, train_test_split
from telco_churn.modeling.run_id import make_run_id
from telco_churn.modeling.metrics.report import project_metric_report
from telco_churn.io.hf import download_dataset_hf, upload_model_bundle
from telco_churn.modeling.dataframe.train_to_df import parquet_to_df
from telco_churn.modeling.bundle.write_bundle import write_bundle
from telco_churn.modeling.optuna import tune_optuna_cv
from telco_churn.modeling.fit import fit_best
from telco_churn.modeling.evaluate import evaluate
from telco_churn.modeling.trainers.make_trainer import make_trainer, available_trainers
from telco_churn.modeling.bundle.model_artifact import ModelArtifact
from telco_churn.logging_utils import setup_logging

from telco_churn.config import (
    REPO_ID,
    REVISION,
    TRAIN_HF_PATH,
    CURRENT_ARTIFACT_VERSION
)

from telco_churn.modeling.config import (
    TARGET_COL,
    PRIMARY_METRIC,
    METRIC_DIRECTION,
    DEFAULT_THRESHOLD,
    HOLDOUT_SIZE,
    SEED,
    CV_SPLITS
)

log = logging.getLogger(__name__)

N_TRIALS = 10
REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_RUNS_DIR = REPO_ROOT / "artifacts" / "runs"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-type",
        dest="model_type",
        required=True,
        choices=available_trainers(),
        help="Which trainer/model family to use",
    )
    p.add_argument("--upload", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()

def main(*, modeltype: str, upload: bool = False) -> None:
    """Train the specified model type and optionally upload the resulting run bundle."""
    t0 = perf_counter()
    run_id = make_run_id()

    log.info(
        "train start run_id=%s model_type=%s upload=%s repo=%s revision=%s train_path=%s",
        run_id, modeltype, upload, REPO_ID, REVISION, TRAIN_HF_PATH,
    )

    metrics = project_metric_report()
    if PRIMARY_METRIC not in metrics:
        raise ValueError(
            f"Unknown primary_metric {PRIMARY_METRIC!r}. Options: {sorted(metrics.keys())}"
        )
    primary_metric_fn = metrics[PRIMARY_METRIC]

    trainer = make_trainer(modeltype, seed=SEED)
    log.info("trainer ready: %s", modeltype)

    log.info("downloading train parquet from HF: repo=%s revision=%s file=%s", REPO_ID, REVISION, TRAIN_HF_PATH)
    local_path = download_dataset_hf(repo_id=REPO_ID, filename=TRAIN_HF_PATH, revision=REVISION)
    log.info("downloaded train parquet: %s", local_path)

    df = parquet_to_df(local_path)
    log.info("loaded dataframe rows=%s cols=%s", len(df), len(df.columns))

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    log.info("split X shape=%s y len=%s target=%s", X.shape, len(y), TARGET_COL)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y,
        test_size=HOLDOUT_SIZE,
        random_state=SEED,
        stratify=y,
    )
    log.info(
        "train/holdout split holdout_size=%.3f train_n=%s holdout_n=%s",
        HOLDOUT_SIZE, len(X_train), len(X_holdout),
    )

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    log.info("cv configured splits=%s seed=%s", CV_SPLITS, SEED)

    log.info("optuna tuning start n_trials=%s primary_metric=%s direction=%s", N_TRIALS, PRIMARY_METRIC, METRIC_DIRECTION)
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
    log.info("optuna tuning done best_params_keys=%s", sorted(best_params.keys()))

    log.info("fit best model")
    artifact = fit_best(
        build_pipeline=trainer.build_pipeline,
        X=X_train,
        y=y_train,
        best_params=best_params,
    )

    log.info("evaluate holdout threshold=%.3f", DEFAULT_THRESHOLD)
    holdout_metrics = evaluate(
        artifact,
        X_holdout,
        y_holdout,
        metrics=metrics,
        threshold=DEFAULT_THRESHOLD,
    )
    log.info("holdout metrics keys=%s", sorted(holdout_metrics.keys()))
    if PRIMARY_METRIC in holdout_metrics:
        log.info("holdout %s=%.6f", PRIMARY_METRIC, float(holdout_metrics[PRIMARY_METRIC]))

    artifact_obj = ModelArtifact(
        run_id=run_id,
        artifact_version=CURRENT_ARTIFACT_VERSION,
        model_type=modeltype,
        model=artifact,
        threshold=DEFAULT_THRESHOLD,
    )

    bundle_dir = ARTIFACT_RUNS_DIR / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    log.info("writing bundle: %s", bundle_dir)

    cfg = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "train_hf_path": TRAIN_HF_PATH,
        "target_col": TARGET_COL,
        "primary_metric": PRIMARY_METRIC,
        "direction": METRIC_DIRECTION,
        "threshold": DEFAULT_THRESHOLD,
        "seed": SEED,
        "holdout_size": HOLDOUT_SIZE,
        "cv_splits": CV_SPLITS,
        "n_trials": N_TRIALS,
    }

    write_bundle(
        bundle_dir=bundle_dir,
        artifact_version=CURRENT_ARTIFACT_VERSION,
        artifact_obj=artifact_obj,
        best_params=best_params,
        cv_summary=cv_summary,
        holdout_metrics=holdout_metrics,
        primary_metric=PRIMARY_METRIC,
        direction=METRIC_DIRECTION,
        cfg=cfg,
    )

    required = ["model.joblib", "metrics.json", "metadata.json"]
    missing = [name for name in required if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle missing required files: {missing} in {bundle_dir}")
    log.info("bundle complete files_ok=%s", required)

    if upload:
        log.info("upload bundle start: repo=%s revision=%s run_id=%s", REPO_ID, REVISION, run_id)
        upload_model_bundle(bundle_dir=bundle_dir, repo_id=REPO_ID, run_id=run_id, revision=REVISION)
        log.info("upload bundle done")

    log.info("train done run_id=%s in %.2fs", run_id, perf_counter() - t0)


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.log_level)
    # Silence sklearn warning caused by LightGBM seeing DataFrame feature names during fit
    # and NumPy arrays during CV/eval. Safe for this project because column order is stable.
    warnings.filterwarnings(
        "ignore",
        message=r"X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        category=UserWarning,
    )
    try:
        main(modeltype=args.model_type, upload=args.upload)
    except Exception:
        log.exception("train failed")
        raise