"""Decide whether to promote a contender by comparing its holdout primary metric to the current champion with an epsilon threshold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    primary_metric: str
    contender_primary: float
    champion_primary: Optional[float] = None
    diff: Optional[float] = None


def _primary_metric_name(m: Dict[str, Any]) -> str:
    pm = m.get("primary_metric")
    if not pm:
        raise ValueError("metrics missing required field: primary_metric")
    return str(pm)


def _holdout_primary_value(m: Dict[str, Any], pm: str) -> float:
    v = m.get("holdout", {}).get(pm)
    if v is None:
        raise ValueError(f"metrics missing holdout primary value for '{pm}'")
    return float(v)

def _artifact_version(m: Optional[Dict[str, Any]]) -> Optional[int]:
    """Return artifact_version if present and parseable; else None."""
    if not m:
        return None
    v = m.get("artifact_version")
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None

def decide_promotion(
    *,
    contender_metrics: Dict[str, Any],
    champion_metrics: Optional[Dict[str, Any]],
    epsilon: float = 1e-3,
) -> PromotionDecision:
    pm = _primary_metric_name(contender_metrics)
    c_val = _holdout_primary_value(contender_metrics, pm)

    if champion_metrics is None:
        return PromotionDecision(
            promote=True,
            reason="No current champion (bootstrap)",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=None,
            diff=None,
        )

    cont_ver = _artifact_version(contender_metrics)
    champ_ver = _artifact_version(champion_metrics)  # missing => None
    if cont_ver != champ_ver:
        return PromotionDecision(
            promote=True,
            reason=f"Artifact version change: champion={champ_ver} contender={cont_ver}",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=None,  # optional: you can still include below, but not necessary
            diff=None,
        )
    
    champ_pm = _primary_metric_name(champion_metrics)
    if champ_pm != pm:
        raise ValueError(f"primary_metric mismatch: contender={pm!r} champion={champ_pm!r}")

    ch_val = _holdout_primary_value(champion_metrics, pm)
    diff = c_val - ch_val

    if diff > epsilon:
        return PromotionDecision(
            promote=True,
            reason=f"Contender improves holdout {pm} by {diff:.6f} (> {epsilon})",
            primary_metric=pm,
            contender_primary=c_val,
            champion_primary=ch_val,
            diff=diff,
        )

    return PromotionDecision(
        promote=False,
        reason=f"Tie/close: contender did not beat champion by > {epsilon} (diff={diff:.6f})",
        primary_metric=pm,
        contender_primary=c_val,
        champion_primary=ch_val,
        diff=diff,
    )
