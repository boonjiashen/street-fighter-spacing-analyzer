import numpy as np
import logging
from scipy import linalg
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator


class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):

        #X = array2d(X)
        X = as_float_array(X, copy = self.copy)
        cov = np.dot(X.T,X) / X.shape[1]
        V, D, _ = np.linalg.svd(cov)  # eigenvalues, eigenvectors

        # new_x = V * (D + epsilon * I)^.5 * V.T * x
        # where (.) is a diagonal matrix and ^.5 is presumably an element-wise
        # operation, x is a column vector
        self.components_  = np.dot(np.dot(V, np.diag((D + self.regularization)**-.5)),
                V.T)

        return self

    def transform(self, X):
        #X = array2d(X)
        #X_transformed = X - self.mean_
        #X_transformed = np.dot(X_transformed, self.components_.T)
        #return X_transformed
        return np.dot(X, self.components_.T)

    def inverse_transform(self, X):
        """Find original X from transformed X"""
        return np.linalg.solve(self.components_, X.T).T
