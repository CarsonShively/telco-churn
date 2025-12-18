import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, log_loss, brier_score_loss,
)

def _get_positive_proba(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 else np.asarray(proba).ravel()
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X)).ravel()
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError("Estimator must support predict_proba or decision_function.")

def _pred_from_proba(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    return (y_proba >= threshold).astype(int)

def average_precision(estimator, X, y, threshold: float = 0.5) -> float:
    y_proba = _get_positive_proba(estimator, X)
    return float(average_precision_score(y, y_proba))

def roc_auc(estimator, X, y, threshold: float = 0.5) -> float:
    y_proba = _get_positive_proba(estimator, X)
    return float(roc_auc_score(y, y_proba))

def f1(estimator, X, y, threshold: float = 0.5) -> float:
    y_proba = _get_positive_proba(estimator, X)
    y_pred = _pred_from_proba(y_proba, threshold)
    return float(f1_score(y, y_pred))

def precision(estimator, X, y, threshold: float = 0.5) -> float:
    y_proba = _get_positive_proba(estimator, X)
    y_pred = _pred_from_proba(y_proba, threshold)
    return float(precision_score(y, y_pred, zero_division=0))

def recall(estimator, X, y, threshold: float = 0.5) -> float:
    y_proba = _get_positive_proba(estimator, X)
    y_pred = _pred_from_proba(y_proba, threshold)
    return float(recall_score(y, y_pred))

def balanced_accuracy(estimator, X, y, threshold: float = 0.5) -> float:
    y_proba = _get_positive_proba(estimator, X)
    y_pred = _pred_from_proba(y_proba, threshold)
    return float(balanced_accuracy_score(y, y_pred))

def neg_log_loss(estimator, X, y, threshold: float = 0.5) -> float:
    y_proba = _get_positive_proba(estimator, X)
    return -float(log_loss(y, y_proba, eps=1e-15))

def neg_brier(estimator, X, y, threshold: float = 0.5) -> float:
    y_proba = _get_positive_proba(estimator, X)
    return -float(brier_score_loss(y, y_proba))

METRICS = {
    "average_precision": average_precision,
    "roc_auc": roc_auc,
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "balanced_accuracy": balanced_accuracy,
    "neg_log_loss": neg_log_loss,
    "neg_brier": neg_brier,
}