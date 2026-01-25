import dagster as dg
from typing import Optional, Any
from telco_churn.promotion.type import RunRow, PromotionDecision
from telco_churn.promotion.decision import decide_promotion

@dg.asset(name="promotion_decision")
def promotion_decision(
    context: dg.AssetExecutionContext,
    champion: Optional[dict[str, Any]],
    best_contender: RunRow,
) -> PromotionDecision:
    contender_metrics = best_contender.metrics
    decision = decide_promotion(
        contender_metrics=contender_metrics,
        champion_metrics=champion,
        epsilon=0.001,
    )

    context.add_output_metadata({
        "promote": decision.promote,
        "reason": decision.reason,
        "primary_metric": decision.primary_metric,
        "contender_primary": decision.contender_primary,
        "champion_primary": decision.champion_primary,
        "diff": decision.diff,
    })

    return decision
