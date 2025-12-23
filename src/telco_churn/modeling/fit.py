"""Fit function helper for train pipeline."""

from typing import Callable, Dict, Any
from sklearn.pipeline import Pipeline

MetricFn = Callable[[Pipeline, Any, Any, float], float]

def fit_best(*, build_pipeline: Callable[[], Pipeline], X, y, best_params: dict) -> Pipeline:
    pipe = build_pipeline()
    pipe.set_params(**best_params)
    pipe.fit(X, y)
    return pipe