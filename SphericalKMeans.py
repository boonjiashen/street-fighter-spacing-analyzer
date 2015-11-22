import numpy as np
import logging
from scipy import linalg
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator


class SphericalKMeans(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters=8, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def init_cluster_centers(self, feature_len):
        """Initialize the dictionary as a k x n matrix
        
        Each row is an atom in the dictionary
        k = n_clusters
        n = length of a feature vector
        """
        k, n  = self.n_clusters, feature_len
        self.cluster_centers_ = np.random.normal(size=(k, n))

        # Normalize so that each atom is of unit length
        self.cluster_centers_ = (self.cluster_centers_.T /  \
                np.sum(self.cluster_centers_, axis=1)).T


    def fit(self, X, y=None):
        """

        `X` = m x n matrix
        `D` = k x n matrix
        `S` = m x k matrix
        `k` = n_clusters
        `n` = length of a feature vector
        `m` = number of examples
        """

        m, n = X.shape
        k = self.n_clusters
        self.init_cluster_centers(n)
        D = self.cluster_centers_

        for _ in range(self.max_iter):
            S = np.dot(X, D.T)
            for s in S:
                label = np.argmax(np.abs(s))
                s[[i for i in range(k) if i != label]] = 0
            D = np.dot(S.T, X) + D
            D = (D.T / np.linalg.norm(D, axis=1)).T

        self.cluster_centers_ = D

        return self

    def transform(self, X):
        S = np.dot(X, self.cluster_centers_.T)
        return S

    def predict(self, X):
        S = np.dot(X, self.cluster_centers_.T)
        labels = np.abs(S).argmax(axis=1)

        return labels

def main():

    kmeans = SphericalKMeans(n_clusters=3)
    kmeans.init_cluster_centers(5)
    n, m = 4, 5
    X = np.random.randint(0, 5, (m, n))
    kmeans.fit(X)
    print(kmeans.cluster_centers_)
    X_test = np.random.randint(0, 5, (100, n))
    print(kmeans.predict(X_test))
    assert False

if __name__ == "__main__":
    main()
