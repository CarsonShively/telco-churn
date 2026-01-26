"""Trainer registry and factory for modeling pipelines."""

from __future__ import annotations

from typing import Dict, Type

from telco_churn.modeling.trainers.lr_trainer import LRTrainer
from telco_churn.modeling.trainers.xgb_trainer import XGBTrainer
from telco_churn.modeling.trainers.lgb_trainer import LGBTrainer


TRAINERS: Dict[str, Type] = {
    "lr": LRTrainer,
    "xgb": XGBTrainer,
    "lgb": LGBTrainer,
}

def available_trainers() -> list[str]:
    return sorted(TRAINERS.keys())


def make_trainer(modeltype: str, *, seed: int):
    try:
        cls = TRAINERS[modeltype]
    except KeyError:
        raise ValueError(f"Unknown modeltype '{modeltype}'. Options: {available_trainers()}")
    return cls(seed=seed)
