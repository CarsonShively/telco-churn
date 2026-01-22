from dataclasses import dataclass
from typing import Any, Optional
import dagster as dg

@dataclass(frozen=True)
class RunRow:
    run_id: str
    model_type: Optional[str]
    metrics: dict[str, Any]
    metrics_path: str
    error: Optional[str] = None

@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    primary_metric: str
    contender_primary: float
    champion_primary: Optional[float] = None
    diff: Optional[float] = None
    
@dataclass(frozen=True)
class ChampionRef:
    run_id: str
    path_in_repo: str
    
class PromotionConfig(dg.Config):
    upload: bool = False