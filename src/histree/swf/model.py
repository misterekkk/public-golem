from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from . import core
from .preprocessing import BinMapper
from .tree import HistogramBasedDecisionTree


class HistogramSequentialWeightedForestRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        reg_lambda: Union[float, int] = 1.0,
        max_features: Union[str, int, float] = 1.0,
        n_bins: int = 256,
        min_gain_to_split: Union[float, int] = 0.0,
        subsample: Union[float, int] = 1.0,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.reg_lambda = reg_lambda
        self.min_gain_to_split = min_gain_to_split
        self.n_bins = n_bins
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):

        X, y = check_X_y(X, y, dtype=np.float32, force_all_finite=True)

        self.n_features_in_ = X.shape[1]

        rng = check_random_state(self.random_state)

        seeds = rng.randint(0, 2**32 - 1, self.n_estimators, dtype=np.uint32)

        self.bin_mapper_ = BinMapper(self.n_bins)
        X_binned = self.bin_mapper_.fit_transform(X, seeds[0])

        n_samples = X.shape[0]

        self.trees_ = []
        self.betas_ = []

        weights = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):

            indices = self._subsample(X, seeds[i])

            X_curr = X_binned[indices]
            y_curr = y[indices]
            w_curr = weights[indices]

            if i > 0:
                X_curr, y_curr, w_curr = self._weight_trimming(
                    X_curr, y_curr, w_curr, seeds[i]
                )

            tree = HistogramBasedDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                reg_lambda=self.reg_lambda,
                min_gain_to_split=self.min_gain_to_split,
                max_features=self.max_features,
                n_bins=self.n_bins,
            )

            tree.fit(X_curr, y_curr, w_curr, seeds[i])

            predictions = tree.predict(X_binned)

            errors = np.abs(predictions - y)
            max_error = np.max(errors)

            if max_error < 1e-9:
                self.trees_.append(tree)
                self.betas_.append(10.0)
                break

            normalized_errors = errors / np.max(errors)
            average_loss = np.sum(weights * normalized_errors)

            if average_loss >= 0.5:
                continue

            self.trees_.append(tree)

            beta = average_loss / (1 - average_loss)
            self.betas_.append(np.log(1 / beta))

            weights_new = weights * np.power(beta, 1.0 - normalized_errors)
            weights_new /= np.sum(weights_new)
            weights = weights_new

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:

        check_is_fitted(self, attributes=["trees_", "betas_", "bin_mapper_"])

        X = check_array(X, dtype=np.float32, force_all_finite=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Model trained on {self.n_features_in_} features, got {X.shape[1]}"
            )

        X_binned = self.bin_mapper_.transform(X)

        max_nodes = 0
        for tree in self.trees_:
            if tree.tree_structure.shape[0] > max_nodes:
                max_nodes = tree.tree_structure.shape[0]

        n_trees = len(self.trees_)
        n_node_features = self.trees_[0].tree_structure.shape[1]

        forest = np.zeros((n_trees, max_nodes, n_node_features), dtype=np.float32)

        for i, tree in enumerate(self.trees_):
            current_nodes = tree.tree_structure.shape[0]
            forest[i, :current_nodes, :] = tree.tree_structure

        predictions = core.predict_forest(X_binned, forest, np.asarray(self.betas_))

        return predictions

    def _subsample(self, X, seed):

        if self.subsample >= 1.0:
            return np.arange(X.shape[0])

        n_samples = X.shape[0]
        size = int(n_samples * self.subsample)

        rng = np.random.default_rng(seed)

        indices = rng.choice(n_samples, size=size, replace=False)

        return indices

    def _weight_trimming(self, X, y, w, seed):

        n_samples = X.shape[0]
        if n_samples <= 50:
            return X, y, w

        top_rate = 0.2
        bottom_rate = 0.1

        top_n = int(n_samples * top_rate)
        bottom_n = int(n_samples * bottom_rate)

        partitioned_indices = np.argpartition(w, -top_n)

        rng = np.random.default_rng(seed)

        top_indices = partitioned_indices[-top_n:]

        bottom_part = partitioned_indices[:-top_n]

        if len(bottom_part) < bottom_n:
            bottom_indices = bottom_part

        else:
            bottom_indices = rng.choice(bottom_part, bottom_n, replace=False)

        multiplier = (1 - top_rate) / bottom_rate

        indices = np.concatenate((top_indices, bottom_indices))

        X_trimmed = X[indices]
        y_trimmed = y[indices]
        w_trimmed = w[indices]

        w_trimmed[top_n:] *= multiplier

        return X_trimmed, y_trimmed, w_trimmed
