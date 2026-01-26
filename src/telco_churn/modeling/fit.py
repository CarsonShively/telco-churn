"""
Fit model with best params.
"""

from typing import Callable, Any
import numpy as np
from sklearn.pipeline import Pipeline

def fit_best(
    *,
    build_pipeline: Callable[[], Pipeline],
    X,
    y,
    best_params: dict,
) -> tuple[Pipeline, list[str]]:
    pipe = build_pipeline()
    pipe.set_params(**best_params)
    pipe.fit(X, y)

    spec = pipe.named_steps["spec"]
    pre = pipe.named_steps["pre"]

    X1 = spec.transform(X.iloc[:1])
    if not hasattr(X1, "columns"):
        raise TypeError("spec.transform(...) must return a DataFrame to extract feature names")

    in_names = np.asarray(X1.columns, dtype=object)
    feature_names = pre.get_feature_names_out(in_names).tolist()

    Xt = pre.transform(X1)
    if Xt.shape[1] != len(feature_names):
        raise ValueError(f"Feature name mismatch: Xt has {Xt.shape[1]} cols, names has {len(feature_names)}")

    return pipe, feature_names
