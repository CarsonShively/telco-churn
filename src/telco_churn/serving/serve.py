"""Serving utilities to load the current champion model, connect to the Redis feature store, and generate churn predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from telco_churn.redis.connect import redis_config, connect_redis
from telco_churn.redis.reader import RedisFeatureStore
from telco_churn.io.hf import read_model_json
from telco_churn.io.hf import load_model_hf
from telco_churn.config import REPO_ID, REVISION

@dataclass
class ServingService:
    model: Any
    fs: RedisFeatureStore
    threshold: float = 0.65
    model_run_id: str | None = None

    @classmethod
    def start(cls) -> "ServingService":
        champion_ptr = read_model_json(repo_id=REPO_ID, revision=REVISION, path_in_repo="champion.json")
        if champion_ptr is None:
            raise RuntimeError("No champion.json found")
        artifact = load_model_hf(
            repo_id=REPO_ID,
            revision=REVISION,
            path_in_repo=f'{champion_ptr["path_in_repo"]}/model.joblib',
        )
        model = getattr(artifact, "model", artifact)

        cfg = redis_config()
        r = connect_redis(cfg)
        fs = RedisFeatureStore(r=r, cfg=cfg)

        return cls(model=model, fs=fs, model_run_id=champion_ptr.get("run_id"),)

    def predict_customer(self, customer_id: str) -> Dict[str, Any]:
        feats = self.fs.fetch_entity_features(customer_id)
        run_prefix = self.fs.current_run_prefix()

        X = pd.DataFrame([feats])

        churn_score: float | None = None
        if hasattr(self.model, "predict_proba"):
            churn_score = float(self.model.predict_proba(X)[0, 1])

        decision_target: bool | None
        if churn_score is not None:
            decision_target = churn_score >= self.threshold
        else:
            pred = self.model.predict(X)[0]
            decision_target = bool(pred) if isinstance(pred, (bool, int)) else None

        return {
            "customer_id": customer_id,
            "feature_store_prefix": run_prefix,
            "churn_score": churn_score,
            "decision_target": decision_target,
            "policy": {"type": "fixed_threshold", "threshold": self.threshold},
        } 