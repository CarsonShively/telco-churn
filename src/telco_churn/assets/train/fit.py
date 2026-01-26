import dagster as dg

from telco_churn.modeling.types import TTSCV, TuningResult, FitOut
from telco_churn.modeling.trainers.make_trainer import make_trainer
from telco_churn.modeling.fit import fit_best
from telco_churn.modeling.config import SEED

@dg.asset(name="fit_pipeline", required_resource_keys={"train_cfg"})
def fit_pipeline(context: dg.AssetExecutionContext, data_splits: TTSCV, best_hyperparameters: TuningResult) -> FitOut:
    """Fit model on best hyperparameters."""
    cfg = context.resources.train_cfg
    trainer = make_trainer(cfg.model_type.value, seed=SEED)

    artifact, feature_names = fit_best(
        build_pipeline=trainer.build_pipeline,
        X=data_splits.X_train,
        y=data_splits.y_train,
        best_params=best_hyperparameters.best_params,
    )

    context.add_output_metadata({
        "model_type": str(cfg.model_type.value),
        "seed": int(SEED),
        "train_rows": int(data_splits.X_train.shape[0]),
        "train_cols": int(data_splits.X_train.shape[1]),
        "best_params": best_hyperparameters.best_params,
        "feature_count": int(len(feature_names)),
        "feature_names_preview": list(feature_names[:10]),
    })

    return FitOut(artifact=artifact, feature_names=feature_names)
