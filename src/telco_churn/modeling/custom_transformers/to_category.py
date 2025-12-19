# switch lgb back to df type category
# figure out array vs df, sparse vs dense, feature names and output format
# look at actual gold features and figure out spec and prerpocess correctness

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ToCategory(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.astype("category")
        else:
            X_df = pd.DataFrame(X)
            return X_df.astype("category")
