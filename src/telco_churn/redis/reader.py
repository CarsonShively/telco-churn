from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import random
import redis

from telco_churn.redis.config import RedisConfig, make_entity_key


@dataclass(frozen=True)
class RedisFeatureStore:
    r: redis.Redis
    cfg: RedisConfig

    def current_run_prefix(self) -> str:
        run_prefix = self.r.get(self.cfg.current_pointer_key)
        if not run_prefix:
            raise RuntimeError(f"Missing CURRENT pointer: {self.cfg.current_pointer_key}")
        if isinstance(run_prefix, (bytes, bytearray)):
            run_prefix = run_prefix.decode("utf-8")
        return str(run_prefix)

    def fetch_entity_features(self, entity: str) -> Dict[str, str]:
        run_prefix = self.current_run_prefix()
        key = make_entity_key(run_prefix, entity)
        feats = self.r.hgetall(key)

        if not feats:
            raise KeyError(f"No features found for entity={entity} (key={key})")

        out: Dict[str, str] = {}
        for k, v in feats.items():
            ks = k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
            vs = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
            out[ks] = vs
        return out

    def sample_entity_ids(self, *, limit: int = 30) -> list[str]:
        prefix = self.current_run_prefix()
        if not prefix.endswith(":"):
            prefix += ":"
        key = f"{prefix}customers:index"

        batch = self.r.srandmember(key, number=limit)
        if not batch:
            return []
        return [
            b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b)
            for b in batch
        ]
