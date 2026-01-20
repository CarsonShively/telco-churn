import dagster as dg
from telco_churn.paths import REPO_ROOT
from telco_churn.modeling.types import FitOut, TuningResult, TrainConfig
from telco_churn.modeling.run_id import make_run_id
from telco_churn.modeling.bundle.model_artifact import ModelArtifact
from telco_churn.modeling.bundle.write_bundle import write_bundle
from telco_churn.modeling.types import BundleOut
from telco_churn.modeling.config import (
    TARGET_COL, PRIMARY_METRIC, METRIC_DIRECTION, HOLDOUT_SIZE, CV_SPLITS, SEED,
)
from telco_churn.config import CURRENT_ARTIFACT_VERSION

@dg.asset(name="bundle")
def bundle(
    fit: FitOut,
    threshold: float,
    holdout: dict[str, float],
    best_params: TuningResult,
    config: TrainConfig,
) -> BundleOut:
    run_id = make_run_id()

    artifact_obj = ModelArtifact(
        run_id=run_id,
        artifact_version=CURRENT_ARTIFACT_VERSION,
        model_type=config.model_type.value,
        model=fit.artifact,
        threshold=threshold,
    )

    bundle_dir = REPO_ROOT / "artifacts" / "runs" / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "target_col": TARGET_COL,
        "primary_metric": PRIMARY_METRIC,
        "direction": METRIC_DIRECTION,
        "threshold": threshold,
        "seed": SEED,
        "holdout_size": HOLDOUT_SIZE,
        "cv_splits": CV_SPLITS,
        "n_trials": config.n_trials,
        "model_type": config.model_type.value,
        "artifact_version": CURRENT_ARTIFACT_VERSION,
    }

    write_bundle(
        bundle_dir=bundle_dir,
        artifact_version=CURRENT_ARTIFACT_VERSION,
        artifact_obj=artifact_obj,
        best_params=best_params.best_params,
        cv_summary=best_params.cv_summary,
        holdout_metrics=holdout,
        primary_metric=PRIMARY_METRIC,
        direction=METRIC_DIRECTION,
        cfg=cfg,
        feature_names=fit.feature_names,
    )

    required = ["model.joblib", "metrics.json", "metadata.json"]
    missing = [name for name in required if not (bundle_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Bundle missing required files: {missing} in {bundle_dir}")
    
    return BundleOut(run_id=run_id, bundle_dir=bundle_dir)