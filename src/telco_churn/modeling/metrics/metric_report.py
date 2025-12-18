from telco_churn.modeling.metrics.metrics import METRICS

def project_metric_report() -> dict:
    return {
        "average_precision": METRICS["average_precision"],
        "roc_auc": METRICS["roc_auc"],
        "f1": METRICS["f1"],
        "precision": METRICS["precision"],
        "recall": METRICS["recall"],
        "neg_brier": METRICS["neg_brier"],
    }
