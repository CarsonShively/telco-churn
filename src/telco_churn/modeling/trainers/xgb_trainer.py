from __future__ import annotations

from dataclasses import dataclass, field

import optuna
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from telco_churn.modeling.feature_spec.feature_spec import FeatureSpecTransformer
from telco_churn.modeling.feature_spec.load_spec import load_feature_spec
from telco_churn.modeling.preprocessors.tree import preprocessor


@dataclass(slots=True)
class XGBTrainer:
    """XGBoost trainer implementation."""
    seed: int = 42
    spec: dict = field(default_factory=load_feature_spec)

    def build_pipeline(self) -> Pipeline:
        id_col = self.spec.get("entity_key", "customer_id")
        return Pipeline(
            steps=[
                ("spec", FeatureSpecTransformer(self.spec, drop_columns=[id_col])),
                ("pre", preprocessor()),
                ("clf", XGBClassifier(
                    random_state=self.seed,
                    tree_method="hist",
                    eval_metric="logloss",
                )),
            ]
        )

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "clf__n_estimators": trial.suggest_int("clf__n_estimators", 300, 2000),
            "clf__learning_rate": trial.suggest_float("clf__learning_rate", 0.01, 0.2, log=True),
            "clf__grow_policy": trial.suggest_categorical("clf__grow_policy", ["depthwise", "lossguide"]),
            "clf__max_depth": trial.suggest_int("clf__max_depth", 3, 8),
            "clf__max_leaves": trial.suggest_int("clf__max_leaves", 16, 256),
            "clf__min_child_weight": trial.suggest_float("clf__min_child_weight", 1.0, 20.0, log=True),
            "clf__subsample": trial.suggest_float("clf__subsample", 0.6, 1.0),
            "clf__colsample_bytree": trial.suggest_float("clf__colsample_bytree", 0.6, 1.0),
            "clf__reg_lambda": trial.suggest_float("clf__reg_lambda", 1e-3, 100.0, log=True),
            "clf__reg_alpha": trial.suggest_float("clf__reg_alpha", 1e-3, 100.0, log=True),
            "clf__gamma": trial.suggest_float("clf__gamma", 0.0, 5.0),
            "clf__scale_pos_weight": trial.suggest_float("clf__scale_pos_weight", 0.5, 20.0, log=True),
        }
