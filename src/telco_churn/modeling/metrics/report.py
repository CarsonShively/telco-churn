from collections.abc import Callable
from telco_churn.modeling.metrics.registry import METRICS

def project_metric_report() -> dict[str, Callable]:
    """Return the set of supported project metrics."""
    keys = [
        "average_precision",
        "roc_auc",
        "f1",
        "precision",
        "recall",
        "neg_brier",
    ]
    return {k: METRICS[k] for k in keys}