import dagster as dg
from telco_churn.promotion.best_candidate import get_best_contender
from telco_churn.promotion.type import RunRow

@dg.asset(name="best_contender")
def best_contender(run_metrics: list[RunRow]) -> RunRow:
    return get_best_contender(run_metrics)