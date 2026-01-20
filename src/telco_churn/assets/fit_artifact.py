import dagster as dg

from telco_churn.modeling.types import TTSCV, TuningResult, TrainConfig, FitOut
from telco_churn.modeling.trainers.make_trainer import make_trainer
from telco_churn.modeling.fit import fit_best
from telco_churn.modeling.config import SEED

@dg.asset(name="fit")
def fit_artifact(tts_cv: TTSCV, best_params: TuningResult, config: TrainConfig) -> FitOut:
    trainer = make_trainer(config.model_type.value, seed=SEED)

    artifact, feature_names = fit_best(
        build_pipeline=trainer.build_pipeline,
        X=tts_cv.X_train,
        y=tts_cv.y_train,
        best_params=best_params.best_params,
    )
    return FitOut(artifact=artifact, feature_names=feature_names)