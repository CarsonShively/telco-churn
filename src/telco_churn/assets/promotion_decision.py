import dagster as dg
from typing import Optional, Any
from telco_churn.promotion.type import RunRow, PromotionDecision
from telco_churn.promotion.decision import decide_promotion

@dg.asset(name="promotion_decision")
def promotion_decision(champion: Optional[dict[str, Any]], best_contender: RunRow) -> PromotionDecision:
    contender_metrics = best_contender.metrics
    return decide_promotion(
        contender_metrics=contender_metrics,
        champion_metrics=champion,
        epsilon=0.001,
    )