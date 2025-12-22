from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import redis

from telco_churn.redis.config import RedisConfig, connect_redis
from telco_churn.redis.reader import RedisFeatureStore
from telco_churn.io.load_model import (
    fetch_champion_pointer,
    load_model_from_champion_pointer,
)


@dataclass(frozen=True)
class ServingConfig:
    repo_id: str = "Carson-Shively/telco-churn"
    revision: str = "main"
    repo_type: str = "model"


@dataclass
class ServingService:
    model: Any
    fs: RedisFeatureStore
    serving_cfg: ServingConfig

    @classmethod
    def from_env(cls) -> "ServingService":
        serving_cfg = ServingConfig()

        champion = fetch_champion_pointer(
            repo_id=serving_cfg.repo_id,
            repo_type=serving_cfg.repo_type,
            revision=serving_cfg.revision,
        )
        if champion is None:
            raise RuntimeError("No champion.json found yet")

        artifact = load_model_from_champion_pointer(
            champion,
            repo_id=serving_cfg.repo_id,
            repo_type=serving_cfg.repo_type,
            revision=serving_cfg.revision,
        )

        model = getattr(artifact, "model", artifact)

        redis_cfg = RedisConfig.from_env()
        r: redis.Redis[str] = connect_redis(redis_cfg)
        fs = RedisFeatureStore(r=r, cfg=redis_cfg)

        return cls(model=model, fs=fs, serving_cfg=serving_cfg)

    def predict_customer(self, customer_id: str) -> Dict[str, Any]:
        run_prefix = self.fs.current_run_prefix()
        feats = self.fs.fetch_entity_features(customer_id)

        X = pd.DataFrame([feats])

        churn_score: float | None = None
        if hasattr(self.model, "predict_proba"):
            churn_score = float(self.model.predict_proba(X)[0, 1])

        threshold = 0.65
        policy: dict[str, Any] = {"type": "fixed_threshold", "threshold": threshold}

        decision_target: bool | None = None
        decision_rule: str

        if churn_score is not None:
            decision_target = churn_score >= threshold
            decision_rule = f"churn_score >= {threshold}"
        else:
            pred = self.model.predict(X)[0]
            decision_target = bool(pred) if isinstance(pred, (bool, int)) else None
            decision_rule = "model.predict() (no predict_proba available)"

        return {
            "customer_id": customer_id,
            "feature_store_prefix": run_prefix,
            "churn_score": churn_score,
            "decision_target": decision_target,
            "policy": policy,
        }