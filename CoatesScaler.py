"""Scale dataset according to Coates & Ng 2012
"""

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class CoatesScaler(BaseEstimator, TransformerMixin):

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)
        return self

    def transform(self, X):
        X_transformed = (X - self.mean) / np.sqrt(self.var + 10)
        return X_transformed
