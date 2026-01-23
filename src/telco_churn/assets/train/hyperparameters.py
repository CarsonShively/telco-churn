import dagster as dg
from telco_churn.modeling.metrics.report import project_metric_report
from telco_churn.modeling.config import PRIMARY_METRIC, METRIC_DIRECTION, SEED
from telco_churn.modeling.types import TuningResult, TTSCV, TrainConfig
from telco_churn.modeling.optuna import tune_optuna_cv
from telco_churn.modeling.trainers.make_trainer import make_trainer

@dg.asset(name="best_hyperparameters")
def best_hyperparameters(data_splits: TTSCV, config: TrainConfig) -> TuningResult:
    metrics = project_metric_report()
    primary_metric_fn = metrics[PRIMARY_METRIC]

    X_train = data_splits.X_train
    y_train = data_splits.y_train
    cv = data_splits.cv
    
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
