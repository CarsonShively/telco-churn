import dagster as dg
import os
from telco_churn.paths import REPO_ROOT
from telco_churn.modeling.types import FitOut, TuningResult
from telco_churn.modeling.run_id import make_run_id
from telco_churn.modeling.bundle.model_artifact import ModelArtifact
from telco_churn.modeling.bundle.write_bundle import write_bundle
from telco_churn.modeling.types import BundleOut
from telco_churn.modeling.config import (
    TARGET_COL, PRIMARY_METRIC, METRIC_DIRECTION, HOLDOUT_SIZE, CV_SPLITS, SEED,
)
from telco_churn.config import CURRENT_ARTIFACT_VERSION

@dg.asset(name="artifact_bundle", required_resource_keys={"train_cfg"})
def artifact_bundle(
    context: dg.AssetExecutionContext, 
    fit_pipeline: FitOut,
    best_threshold: float,
    holdout_evaluation: dict[str, float],
    best_hyperparameters: TuningResult,
) -> BundleOut:
    cfg = context.resources.train_cfg
    run_id = make_run_id()

    artifact_obj = ModelArtifact(
        run_id=run_id,
        artifact_version=CURRENT_ARTIFACT_VERSION,
        model_type=cfg.model_type.value,
        model=fit_pipeline.artifact,
        threshold=best_threshold,
    )

    bundle_dir = REPO_ROOT / "artifacts" / "runs" / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "target_col": TARGET_COL,
        "primary_metric": PRIMARY_METRIC,
        "direction": METRIC_DIRECTION,
        "threshold": best_threshold,
        "seed": SEED,
        "holdout_size": HOLDOUT_SIZE,
        "cv_splits": CV_SPLITS,
        "n_trials": cfg.n_trials,
        "model_type": cfg.model_type.value,
        "artifact_version": CURRENT_ARTIFACT_VERSION,
    }

    write_bundle(
        bundle_dir=bundle_dir,
        artifact_version=CURRENT_ARTIFACT_VERSION,
        artifact_obj=artifact_obj,
        best_params=best_hyperparameters.best_params,
        cv_summary=best_hyperparameters.cv_summary,
        holdout_metrics=holdout_evaluation,
        primary_metric=PRIMARY_METRIC,
        direction=METRIC_DIRECTION,
        cfg=cfg,
        feature_names=fit_pipeline.feature_names,
    )

    required = ["model.joblib", "metrics.json", "metadata.json"]
    missing = [name for name in required if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle missing required files: {missing} in {bundle_dir}")

    context.add_output_metadata({
        "run_id": run_id,
        "artifact_version": int(CURRENT_ARTIFACT_VERSION),
        "model_type": str(cfg["model_type"]),
        "bundle_dir": dg.MetadataValue.path(str(bundle_dir)),
        "threshold": float(best_threshold),
        "primary_metric": str(PRIMARY_METRIC),
        "direction": str(METRIC_DIRECTION),

        "feature_count": int(len(fit_pipeline.feature_names)),
        "feature_names_preview": list(fit_pipeline.feature_names[:10]),

        "required_files": required,
        "model_path": dg.MetadataValue.path(str(bundle_dir / "model.joblib")),
        "metrics_path": dg.MetadataValue.path(str(bundle_dir / "metrics.json")),
        "metadata_path": dg.MetadataValue.path(str(bundle_dir / "metadata.json")),
        "model_bytes": os.path.getsize(bundle_dir / "model.joblib"),
        "metrics_bytes": os.path.getsize(bundle_dir / "metrics.json"),
        "metadata_bytes": os.path.getsize(bundle_dir / "metadata.json"),

        "holdout_metrics": holdout_evaluation,
        "best_params": best_hyperparameters.best_params,
    })

    return BundleOut(run_id=run_id, bundle_dir=bundle_dir)
