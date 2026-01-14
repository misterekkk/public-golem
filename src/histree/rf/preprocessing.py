import numpy as np

from . import core


class BinMapper:
    def __init__(self, n_bins=256, subsample=200000):
        self.n_bins = n_bins
        self.subsample = subsample
        self.bin_thresholds = []

        self.thresholds_matrix = None
        self.counts_vector = None

    def fit(self, X, seed=None):

        n_samples, n_features = X.shape

        if n_samples > self.subsample:

            rng = np.random.default_rng(seed)
            batch = rng.choice(n_samples, size=self.subsample, replace=False)
            X_sample = X[batch]

        else:

            X_sample = X

        for i in range(n_features):

            data = X_sample[:, i]

            percentiles = np.linspace(0, 100, self.n_bins + 1)

            thresholds = np.unique(np.percentile(data, percentiles[1:-1]))

            self.bin_thresholds.append(np.ascontiguousarray(thresholds))

        self._preprocess_numba()

        return self

    def transform(self, X):

        X = np.asarray(X, dtype=np.float32)

        return core.transform_bin_mapper(X, self.thresholds_matrix, self.counts_vector)

    def fit_transform(self, X, seed=None):
        self.fit(X, seed)
        return self.transform(X)

    def _preprocess_numba(self):

        n_features = len(self.bin_thresholds)

        max_n_thresholds = max(len(t) for t in self.bin_thresholds)

        self.thresholds_matrix = np.full(
            (n_features, max_n_thresholds), np.inf, dtype=np.float64
        )

        self.counts_vector = np.zeros(n_features, dtype=np.int32)

        for i, t in enumerate(self.bin_thresholds):
            k = len(t)
            self.thresholds_matrix[i, :k] = t
            self.counts_vector[i] = k
