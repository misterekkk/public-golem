from typing import Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from . import core
from .preprocessing import BinMapper
from .tree import HistogramBasedDecisionTree


class HistogramRandomForestRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = 1.0,
        n_bins: int = 256,
        bootstrap: bool = True,
        max_samples: Optional[Union[int, float]] = None,
        min_gain_to_split: float = 0.0,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_bins = n_bins
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.min_gain_to_split = min_gain_to_split
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):

        X, y = check_X_y(X, y, dtype=np.float32, force_all_finite=True)

        self.n_features_in_ = X.shape[1]

        rng = check_random_state(self.random_state)
        seeds = rng.randint(0, 2**32 - 1, size=self.n_estimators, dtype=np.uint32)

        self.bin_mapper_ = BinMapper(self.n_bins)
        X_binned = self.bin_mapper_.fit_transform(X, seeds[0])

        n_samples = X_binned.shape[0]

        if self.max_samples is None:
            sample_size = n_samples

        elif isinstance(self.max_samples, float):
            sample_size = int(n_samples * self.max_samples)

        else:
            sample_size = self.max_samples

        def _build_one_tree(seed):

            local_rng = np.random.default_rng(seed)

            if self.bootstrap:
                indices = local_rng.choice(n_samples, sample_size, replace=True).astype(
                    np.int64
                )

            else:
                indices = np.arange(n_samples, dtype=np.int64)

            depth = self.max_depth if self.max_depth is not None else -1

            tree = HistogramBasedDecisionTree(
                max_depth=depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_gain_to_split=self.min_gain_to_split,
                max_features=self.max_features,
                n_bins=self.n_bins,
            )

            tree.fit(X_binned, y, indices)

            return tree

        self.trees_ = Parallel(n_jobs=self.n_jobs or -1, prefer="threads")(
            delayed(_build_one_tree)(s) for s in seeds
        )

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:

        check_is_fitted(self, ["trees_", "bin_mapper_"])

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

        return predictions
