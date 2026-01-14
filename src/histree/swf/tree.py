import numpy as np

from . import core


class HistogramBasedDecisionTree:

    def __init__(
        self,
        max_depth=-1,
        min_samples_split=2,
        min_samples_leaf=1,
        reg_lambda=1.0,
        min_gain_to_split=0.0,
        max_features=1.0,
        n_bins=256,
    ):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.min_gain_to_split = min_gain_to_split
        self.max_features = max_features
        self.n_bins = n_bins

        self.tree_structure = None

    def fit(self, X_binned, y, weights, seed=None):

        n_samples = X_binned.shape[0]

        indices = np.arange(n_samples, dtype=np.int64)

        self.n_features = X_binned.shape[1]
        n_features_batch = self._get_n_features()

        rng = np.random.default_rng(seed)

        features_indices = rng.choice(
            self.n_features, n_features_batch, replace=False
        ).astype(np.int32)

        features_indices.sort()

        self.tree_structure = core.build_tree(
            X_binned,
            y,
            weights,
            indices,
            features_indices,
            self.max_depth,
            self.n_bins,
            self.min_samples_split,
            self.min_samples_leaf,
            self.min_gain_to_split,
        )

        return self

    def predict(self, X_binned):

        return core.predict_tree(X_binned, self.tree_structure)

    def _get_n_features(self):
        if self.max_features == "sqrt":
            return int(np.sqrt(self.n_features))

        elif self.max_features == "log2":
            return int(np.log2(self.n_features))

        elif isinstance(self.max_features, int):
            return min(self.max_features, self.n_features)

        elif isinstance(self.max_features, float):
            return int(self.max_features * self.n_features)

        else:
            return self.n_features
