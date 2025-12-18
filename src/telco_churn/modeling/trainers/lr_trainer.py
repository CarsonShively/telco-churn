from __future__ import annotations

from dataclasses import dataclass, field

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from telco_churn.modeling.feature_spec.feature_spec import FeatureSpecTransformer
from telco_churn.modeling.feature_spec.load_spec import load_feature_spec
from telco_churn.modeling.preprocessors import lr_preprocessor


@dataclass(slots=True)
class LRTrainer:
    seed: int = 42
    spec: dict = field(default_factory=load_feature_spec)

    def build_pipeline(self) -> Pipeline:
        id_col = self.spec.get("entity_key", "customer_id")
        return Pipeline(
            steps=[
                ("spec", FeatureSpecTransformer(self.spec, drop_columns=[id_col])),
                ("pre", lr_preprocessor()),
                ("clf", LogisticRegression(max_iter=2000, random_state=self.seed)),
            ]
        )

    def suggest_params(self, trial: optuna.Trial) -> dict:
        solver = "saga"

        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
        l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95) if penalty == "elasticnet" else None

        params = {
            "clf__C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "clf__solver": solver,
            "clf__penalty": penalty,
            "clf__class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

        if l1_ratio is not None:
            params["clf__l1_ratio"] = l1_ratio

        return params
