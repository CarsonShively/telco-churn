import dagster as dg
from telco_churn.modeling.metrics.report import project_metric_report
from telco_churn.modeling.config import PRIMARY_METRIC, METRIC_DIRECTION, SEED
from telco_churn.modeling.types import TuningResult, TTSCV, TrainConfig
from telco_churn.modeling.optuna import tune_optuna_cv
from telco_churn.modeling.trainers.make_trainer import make_trainer

@dg.asset(name="best_params")
def best_params(tts_cv: TTSCV, config: TrainConfig) -> TuningResult:
    metrics = project_metric_report()
    primary_metric_fn = metrics[PRIMARY_METRIC]

    X_train = tts_cv.X_train
    y_train = tts_cv.y_train
    cv = tts_cv.cv
    
    trainer = make_trainer(config.model_type.value, seed=SEED)

    best_params, cv_summary = tune_optuna_cv(
        build_pipeline=trainer.build_pipeline,
        suggest_params=trainer.suggest_params,
        X=X_train,
        y=y_train,
        cv=cv,
        primary_metric=primary_metric_fn,
        metrics=metrics,
        direction=METRIC_DIRECTION,
        n_trials=config.n_trials,
        seed=SEED,
    )

    return TuningResult(best_params=best_params, cv_summary=cv_summary)
