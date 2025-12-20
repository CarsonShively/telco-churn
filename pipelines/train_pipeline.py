import argparse
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

from telco_churn.io.hf import download_from_hf, upload_bundle
from telco_churn.modeling.dataframe.train_to_df import parquet_to_df
from telco_churn.modeling.metrics.metric_report import project_metric_report
from telco_churn.modeling.assemble_bundle.assemble_bundle import write_bundle
from telco_churn.modeling.optuna import tune_optuna_cv
from telco_churn.modeling.train_pipeline_ops import fit_best, evaluate
from telco_churn.modeling.trainers.make_trainer import make_trainer, available_trainers
from telco_churn.env_utils.helpers import env_str, env_int, env_float, env_choice
from telco_churn.modeling.run_id import make_run_id
from telco_churn.modeling.assemble_bundle.assemble_artifact import ModelArtifact

@dataclass(frozen=True)
class TrainPipelineConfig:
    repo_id: str = env_str("TELCO_REPO_ID", "Carson-Shively/telco-churn")
    revision: str = env_str("TELCO_REVISION", "main")
    train_hf_path: str = env_str("TELCO_TRAIN_HF_PATH", "data/gold/train.parquet")

    target_col: str = env_str("TELCO_TARGET_COL", "churn")
    holdout_size: float = env_float("TELCO_HOLDOUT_SIZE", 0.2)

    seed: int = env_int("TELCO_SEED", 42)
    cv_splits: int = env_int("TELCO_CV_SPLITS", 5)
    n_trials: int = env_int("TELCO_N_TRIALS", 50)

    primary_metric: str = env_str("TELCO_PRIMARY_METRIC", "average_precision")
    direction: str = env_choice("TELCO_DIRECTION", "maximize", {"maximize", "minimize"})
    threshold: float = env_float("TELCO_THRESHOLD", 0.5)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--modeltype", required=True, choices=available_trainers())
    p.add_argument(
        "--upload",
        action="store_true",
        help="Upload the run bundle to Hugging Face after training",
    )
    return p.parse_args()


def main(cfg: TrainPipelineConfig, *, modeltype: str, upload: bool = False) -> None:
    run_id = make_run_id()
    
    metrics = project_metric_report()

    if cfg.primary_metric not in metrics:
        raise ValueError(
            f"Unknown primary_metric {cfg.primary_metric!r}. Options: {sorted(metrics.keys())}"
        )

    primary_metric_fn = metrics[cfg.primary_metric]

    trainer = make_trainer(modeltype, seed=cfg.seed)

    local_path = download_from_hf(
        repo_id=cfg.repo_id,
        filename=cfg.train_hf_path,
        revision=cfg.revision,
    )
    df = parquet_to_df(local_path)

    target_col = cfg.target_col
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if not (0.0 < cfg.holdout_size < 1.0):
        raise ValueError(f"holdout_size must be in (0, 1), got {cfg.holdout_size}")

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=cfg.holdout_size, random_state=cfg.seed, stratify=y
)

    cv = StratifiedKFold(n_splits=cfg.cv_splits, shuffle=True, random_state=cfg.seed)

    best_params, cv_summary = tune_optuna_cv(
        build_pipeline=trainer.build_pipeline,
        suggest_params=trainer.suggest_params,
        X=X_train, y=y_train, cv=cv,
        primary_metric=primary_metric_fn,
        metrics=metrics,
        direction=cfg.direction,
        n_trials=cfg.n_trials,
        seed=cfg.seed,
    )

    artifact = fit_best(
        build_pipeline=trainer.build_pipeline,
        X=X_train, y=y_train,
        best_params=best_params,
    )

    holdout_metrics = evaluate(
        artifact, X_holdout, y_holdout,
        metrics=metrics,
        threshold=cfg.threshold,
    )
    
    artifact_obj = ModelArtifact(
        run_id=run_id,
        model_type=modeltype,
        model=artifact,
        threshold=cfg.threshold,
    )
    
    bundle_dir = Path("artifacts/runs") / run_id

    bundle_dir = write_bundle(
        bundle_dir=bundle_dir,
        artifact_obj=artifact_obj,
        best_params=best_params,
        cv_summary=cv_summary,
        holdout_metrics=holdout_metrics,
        primary_metric=cfg.primary_metric,
        direction=cfg.direction,
        cfg=cfg,
    )
    
    required = ["model.joblib", "metrics.json", "metadata.json"]
    missing = [name for name in required if not (Path(bundle_dir) / name).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle missing required files: {missing} in {bundle_dir}")

    if upload:
        try:
            upload_bundle(
                bundle_dir=bundle_dir,
                repo_id=cfg.repo_id,
                run_id=run_id,
                revision=cfg.revision,
            )
        except Exception as e:
            print(f"[WARN] Upload failed: {e}\nBundle is saved locally; you can retry upload later.")
    else:
        print("[INFO] Skipping upload (run bundle saved locally).")

if __name__ == "__main__":
    args = parse_args()
    cfg = TrainPipelineConfig()
    main(cfg, modeltype=args.modeltype, upload=args.upload)