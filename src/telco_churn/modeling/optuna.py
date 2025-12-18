from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import optuna
from sklearn.pipeline import Pipeline

MetricFn = Callable[[Pipeline, Any, Any], float]


def _take_rows(a, idx):
    return a.iloc[idx] if hasattr(a, "iloc") else a[idx]


def tune_optuna_cv(
    *,
    build_pipeline: Callable[[], Pipeline],
    suggest_params: Callable[[optuna.Trial], Dict[str, Any]],
    X: Any,
    y: Any,
    cv,
    primary_metric: MetricFn,
    metrics: Optional[Dict[str, MetricFn]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    n_trials: int = 50,
    direction: str = "maximize",
    seed: int = 42,
    use_pruning: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    metrics = metrics or {"primary": primary_metric}
    fixed_params = fixed_params or {}

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1) if use_pruning else optuna.pruners.NopPruner()
    sampler = optuna.samplers.TPESampler(seed=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed_params, **suggest_params(trial)}

        fold_scores: list[float] = []
        for i, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
            X_tr, X_va = _take_rows(X, tr_idx), _take_rows(X, va_idx)
            y_tr, y_va = _take_rows(y, tr_idx), _take_rows(y, va_idx)

            pipe = build_pipeline()
            pipe.set_params(**params)
            pipe.fit(X_tr, y_tr)

            score = float(primary_metric(pipe, X_va, y_va))
            fold_scores.append(score)

            trial.report(float(np.mean(fold_scores)), step=i)
            if use_pruning and trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = {**fixed_params, **study.best_trial.params}

    per_metric_scores: dict[str, list[float]] = {name: [] for name in metrics}
    for tr_idx, va_idx in cv.split(X, y):
        X_tr, X_va = _take_rows(X, tr_idx), _take_rows(X, va_idx)
        y_tr, y_va = _take_rows(y, tr_idx), _take_rows(y, va_idx)

        pipe = build_pipeline()
        pipe.set_params(**best_params)
        pipe.fit(X_tr, y_tr)

        for name, fn in metrics.items():
            per_metric_scores[name].append(float(fn(pipe, X_va, y_va)))

    cv_summary = {
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "n_trials": int(n_trials),
        "direction": direction,
        "best_params": dict(best_params),
        "metrics": {
            name: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "per_fold": [float(v) for v in vals],
            }
            for name, vals in per_metric_scores.items()
        },
    }

    return best_params, cv_summary
