from __future__ import annotations

from dataclasses import dataclass, field

import optuna
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from telco_churn.modeling.feature_spec.feature_spec import FeatureSpecTransformer
from telco_churn.modeling.feature_spec.load_spec import load_feature_spec
from telco_churn.modeling.preprocessors import lgb_preprocessor  


@dataclass(slots=True)
class LGBTrainer:
    seed: int = 42
    spec: dict = field(default_factory=load_feature_spec)

    def build_pipeline(self) -> Pipeline:
        id_col = self.spec.get("entity_key", "customer_id")
        return Pipeline(
            steps=[
                ("spec", FeatureSpecTransformer(self.spec, drop_columns=[id_col])),
                ("pre", lgb_preprocessor()),
                ("clf", LGBMClassifier(
                    random_state=self.seed,
                    n_jobs=-1,
                    objective="binary",
                    metric="binary_logloss",
                    verbose=-1,
                )),
            ]
        )

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return {
            "clf__n_estimators": trial.suggest_int("n_estimators", 300, 3000),
            "clf__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "clf__num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "clf__max_depth": trial.suggest_int("max_depth", -1, 12), 
            "clf__min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "clf__subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "clf__subsample_freq": trial.suggest_int("subsample_freq", 0, 10),
            "clf__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "clf__reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "clf__reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 100.0, log=True),
            "clf__min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
            "clf__scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 20.0, log=True),
        }
