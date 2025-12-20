from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    contender_primary: Optional[float] = None
    champion_primary: Optional[float] = None
    primary_metric: Optional[str] = None


def _primary_metric_name(m: Dict[str, Any]) -> Optional[str]:
    pm = m.get("primary_metric")
    return pm if pm else None


def _holdout_primary_value(m: Dict[str, Any]) -> Optional[float]:
    pm = _primary_metric_name(m)
    if not pm:
        return None
    return m.get("holdout", {}).get(pm)


def _passes_selection_gates(
    m: Dict[str, Any], *, min_recall: float, min_precision: float
) -> bool:
    pm = _primary_metric_name(m)
    if not pm:
        return False
    h = m.get("holdout", {})
    primary_val = h.get(pm)
    rec = h.get("recall")
    prec = h.get("precision")
    return (
        primary_val is not None
        and rec is not None and rec >= min_recall
        and prec is not None and prec >= min_precision
    )


def _cv_mean(m: Dict[str, Any]) -> Optional[float]:
    pm = _primary_metric_name(m)
    if not pm:
        return None
    return m.get("cv", {}).get("metrics", {}).get(pm, {}).get("mean")


def _cv_std(m: Dict[str, Any]) -> Optional[float]:
    pm = _primary_metric_name(m)
    if not pm:
        return None
    return m.get("cv", {}).get("metrics", {}).get(pm, {}).get("std")


def decide_promotion(
    *,
    contender_metrics: Dict[str, Any],
    champion_metrics: Optional[Dict[str, Any]],
    min_recall: float = 0.20,
    min_precision: float = 0.10,
    min_improvement: float = 0.0,
    tie_break_use_cv: bool = True,
) -> PromotionDecision:
    """
    Decide whether to promote contender -> champion.

    Rules:
      1) Contender must pass gates (primary metric exists + holdout precision/recall thresholds).
      2) If no champion exists: promote (bootstrap).
      3) Otherwise compare holdout primary metric:
           - promote if contender >= champion + min_improvement
           - if within margin and tie_break_use_cv: use CV mean/std to break ties
    """
    pm = _primary_metric_name(contender_metrics)
    if not pm:
        return PromotionDecision(promote=False, reason="Contender missing primary_metric")

    if not _passes_selection_gates(contender_metrics, min_recall=min_recall, min_precision=min_precision):
        return PromotionDecision(
            promote=False,
            reason=f"Contender failed gates (min_recall={min_recall}, min_precision={min_precision})",
            primary_metric=pm,
            contender_primary=_holdout_primary_value(contender_metrics),
            champion_primary=_holdout_primary_value(champion_metrics) if champion_metrics else None,
        )

    c_val = _holdout_primary_value(contender_metrics)
    if c_val is None:
        return PromotionDecision(promote=False, reason="Contender missing holdout primary value", primary_metric=pm)

    if champion_metrics is None:
        return PromotionDecision(
            promote=True,
            reason="No current champion (bootstrap)",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=None,
        )

    champ_pm = _primary_metric_name(champion_metrics)
    if champ_pm != pm:
        return PromotionDecision(
            promote=False,
            reason=f"Champion primary_metric ({champ_pm}) != contender primary_metric ({pm})",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=_holdout_primary_value(champion_metrics),
        )

    ch_val = _holdout_primary_value(champion_metrics)
    if ch_val is None:
        return PromotionDecision(
            promote=True,
            reason="Champion missing holdout primary value (treat as invalid champion)",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=None,
        )

    diff = c_val - ch_val
    if diff >= min_improvement:
        return PromotionDecision(
            promote=True,
            reason=f"Contender beats champion on holdout {pm} by {diff:.6f} (>= {min_improvement})",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=ch_val,
        )

    if not tie_break_use_cv:
        return PromotionDecision(
            promote=False,
            reason=f"Contender did not beat champion by min_improvement (diff={diff:.6f})",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=ch_val,
        )

    c_mean, ch_mean = _cv_mean(contender_metrics), _cv_mean(champion_metrics)
    c_std, ch_std = _cv_std(contender_metrics), _cv_std(champion_metrics)

    if c_mean is None or ch_mean is None:
        return PromotionDecision(
            promote=False,
            reason=f"Within margin and CV mean missing (diff={diff:.6f})",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=ch_val,
        )

    if c_mean > ch_mean:
        return PromotionDecision(
            promote=True,
            reason=f"Within margin; contender wins by higher CV mean ({c_mean:.6f} > {ch_mean:.6f})",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=ch_val,
        )

    if c_mean < ch_mean:
        return PromotionDecision(
            promote=False,
            reason=f"Within margin; champion wins by higher CV mean ({ch_mean:.6f} >= {c_mean:.6f})",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=ch_val,
        )

    if c_std is not None and ch_std is not None and c_std < ch_std:
        return PromotionDecision(
            promote=True,
            reason=f"Within margin; CV mean tie, contender has lower CV std ({c_std:.6f} < {ch_std:.6f})",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=ch_val,
        )

    return PromotionDecision(
        promote=False,
        reason="Within margin; no tie-break advantage for contender",
        primary_metric=pm,
        contender_primary=c_val,
        champion_primary=ch_val,
    )