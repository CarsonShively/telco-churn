import dagster as dg

from telco_churn.modeling.types import TTSCV, TuningResult, FitOut
from telco_churn.modeling.trainers.make_trainer import make_trainer
from telco_churn.modeling.fit import fit_best
from telco_churn.modeling.config import SEED

@dg.asset(name="fit_pipeline", required_resource_keys={"train_cfg"})
def fit_pipeline(context: dg.AssetExecutionContext, data_splits: TTSCV, best_hyperparameters: TuningResult) -> FitOut:
    cfg = context.resources.train_cfg
    trainer = make_trainer(cfg.model_type.value, seed=SEED)

    artifact, feature_names = fit_best(
        build_pipeline=trainer.build_pipeline,
        X=data_splits.X_train,
        y=data_splits.y_train,
        best_params=best_hyperparameters.best_params,
    )
    return FitOut(artifact=artifact, feature_names=feature_names)