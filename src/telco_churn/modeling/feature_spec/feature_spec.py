"""Sklearn Pipeline wrapper around feature_spec."""

from __future__ import annotations
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from telco_churn.modeling.feature_spec.apply import feature_spec


class FeatureSpecTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        spec: Dict[str, Any],
        *,
        add_missing_columns: bool = True,
        drop_columns: Optional[list[str]] = None,
    ):
        self.spec = spec
        self.add_missing_columns = add_missing_columns
        self.drop_columns = drop_columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSpecTransformer expects a pandas DataFrame as input.")

        df = X.copy()

        if self.drop_columns:
            df = df.drop(columns=self.drop_columns, errors="ignore")

        return feature_spec(
            df,
            self.spec,
            add_missing_columns=self.add_missing_columns,
        )
