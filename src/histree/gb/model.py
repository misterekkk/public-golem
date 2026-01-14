from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from . import core
from .preprocessing import BinMapper
from .tree import HistogramBasedDecisionTree


class HistogramGradientBoostingRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 20,
        max_features: float = 1.0,
        reg_lambda: float = 1.0,
        min_gain_to_split: float = 0.0,
        n_bins: int = 256,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.reg_lambda = reg_lambda
        self.min_gain_to_split = min_gain_to_split
        self.n_bins = n_bins
        self.subsample = subsample
        self.random_state = random_state

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame] = None,
        y_val: Union[np.ndarray, pd.Series] = None,
        early_stopping_rounds: Optional[int] = None,
    ):

        X, y = check_X_y(X, y, dtype=np.float32, force_all_finite=True)
        self.n_features_in_ = X.shape[1]

        rng = check_random_state(self.random_state)

        seeds = rng.randint(0, 2**32 - 1, self.n_estimators, dtype=np.uint32)

        self.bin_mapper_ = BinMapper(self.n_bins)
        X_binned = self.bin_mapper_.fit_transform(X, seeds[0])

        X_val_binned = None
        if X_val is not None and y_val is not None:
            X_val, y_val = check_X_y(
                X_val, y_val, dtype=np.float32, force_all_finite=True
            )
            X_val_binned = self.bin_mapper_.transform(X_val)

        self.initial_prediction_ = np.mean(y)
        self.trees_ = []

        F_train = np.full(len(y), self.initial_prediction_, dtype=np.float64)
        F_val = None

        if X_val_binned is not None:
            F_val = np.full(len(y_val), self.initial_prediction_, dtype=np.float64)

        best_val_score = np.inf
        no_improve_count = 0
        self.best_iteration_ = 0

        for i in range(self.n_estimators):

            gradients = F_train - y
            hessians = np.ones_like(y, dtype=np.float32)

            X_sample, g_sample, h_sample = self._subsample(
                X_binned, gradients, hessians, seeds[i]
            )
            X_goss, g_goss, h_goss = self._goss(X_sample, g_sample, h_sample, seeds[i])

            tree = HistogramBasedDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                reg_lambda=self.reg_lambda,
                min_gain_to_split=self.min_gain_to_split,
                n_bins=self.n_bins,
            )

            tree.fit(X_goss, g_goss, h_goss, seeds[i])
            self.trees_.append(tree)

            update_train = tree.predict(X_binned)

            F_train += self.learning_rate * update_train

            if X_val_binned is not None:

                update_val = tree.predict(X_val_binned)
                F_val += self.learning_rate * update_val

                current_val_score = np.mean(np.power(y_val - F_val, 2))

                if current_val_score < best_val_score:

                    best_val_score = current_val_score
                    self.best_iteration_ = i
                    no_improve_count = 0

                else:

                    no_improve_count += 1

                if early_stopping_rounds and no_improve_count >= early_stopping_rounds:

                    self.trees_ = self.trees_[: self.best_iteration_ + 1]
                    break

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:

        check_is_fitted(self, ["trees_", "bin_mapper_", "initial_prediction_"])

        X_arr = check_array(X, dtype=np.float32, force_all_finite=True)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Model trained on {self.n_features_in_} features, got {X_arr.shape[1]}"
            )

        X_binned = self.bin_mapper_.transform(X_arr)

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

        predictions = core.predict_forest(X_binned, forest)

        return predictions * self.learning_rate + self.initial_prediction_

    def _subsample(self, X, g, h, seed):

        if self.subsample >= 1.0:
            return X, g, h

        n_samples = X.shape[0]
        size = int(n_samples * self.subsample)

        rng = np.random.default_rng(seed)

        indices = rng.choice(n_samples, size=size, replace=False)

        return X[indices], g[indices], h[indices]

    def _goss(self, X, g, h, seed):

        n_samples = X.shape[0]
        if n_samples <= 50:
            return X, g, h

        top_rate = 0.2
        bottom_rate = 0.1

        top_n = int(n_samples * top_rate)
        bottom_n = int(n_samples * bottom_rate)
        abs_g = np.abs(g)

        partitioned_indices = np.argpartition(abs_g, -top_n)

        rng = np.random.default_rng(seed)

        top_indices = partitioned_indices[-top_n:]

        bottom_part = partitioned_indices[:-top_n]

        if len(bottom_part) < bottom_n:
            bottom_indices = bottom_part

        else:
            bottom_indices = rng.choice(bottom_part, bottom_n, replace=False)

        multiplier = (1 - top_rate) / bottom_rate

        indices = np.concatenate((top_indices, bottom_indices))

        X_goss = X[indices]
        g_goss = g[indices]
        h_goss = h[indices]

        g_goss[top_n:] *= multiplier
        h_goss[top_n:] *= multiplier

        return X_goss, g_goss, h_goss
