import dagster as dg
from telco_churn.modeling.metrics.report import project_metric_report
from telco_churn.modeling.config import PRIMARY_METRIC, METRIC_DIRECTION, SEED
from telco_churn.modeling.types import TuningResult, TTSCV
from telco_churn.modeling.optuna import tune_optuna_cv
from telco_churn.modeling.trainers.make_trainer import make_trainer

@dg.asset(name="best_hyperparameters", required_resource_keys={"train_cfg"})
def best_hyperparameters(context: dg.AssetExecutionContext, data_splits: TTSCV) -> TuningResult:
    cfg = context.resources.train_cfg
    metrics = project_metric_report()
    primary_metric_fn = metrics[PRIMARY_METRIC]

    X_train = data_splits.X_train
    y_train = data_splits.y_train
    cv = data_splits.cv
    
    trainer = make_trainer(cfg.model_type.value, seed=SEED)

    best_params, cv_summary = tune_optuna_cv(
        build_pipeline=trainer.build_pipeline,
        suggest_params=trainer.suggest_params,
        X=X_train,
        y=y_train,
        cv=cv,
        primary_metric=primary_metric_fn,
        metrics=metrics,
        direction=METRIC_DIRECTION,
        n_trials=cfg.n_trials,
        seed=SEED,
    )

    context.add_output_metadata({
        "model_type": str(cfg.model_type.value),
        "n_trials": int(cfg.n_trials),
        "primary_metric": str(PRIMARY_METRIC),
        "metric_direction": str(METRIC_DIRECTION),
        "seed": int(SEED),

        "best_params": best_params,
        "cv_summary_keys": list(cv_summary.keys()) if hasattr(cv_summary, "keys") else None,
    })

    return TuningResult(best_params=best_params, cv_summary=cv_summary)
