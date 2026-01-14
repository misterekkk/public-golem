import numpy as np
import pytest

import histree.rf.core as rf_core
from histree import HistogramRandomForestRegressor


@pytest.fixture
def multivariate_data():
    """Generates a multivariate dataset with correlated features."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(100, 10)).astype(np.float32)
    # y depends strongly on first two features
    y = 5 * X[:, 0] + 3 * X[:, 1] + rng.normal(0, 0.1, 100).astype(np.float32)
    return X, y


def test_rf_bootstrap_variance(multivariate_data):
    """
    Validates that bootstrap sampling induces variance among base estimators.
    """
    X, y = multivariate_data
    model = HistogramRandomForestRegressor(
        n_estimators=5, bootstrap=True, random_state=42
    )
    model.fit(X, y)

    X_binned = model.bin_mapper_.transform(X)

    # Collect raw predictions from each individual tree
    tree_preds = []
    for tree in model.trees_:
        # FIX: Używamy core.predict_tree zamiast tree.predict
        # Drzewo w RF to tylko kontener na strukturę (numpy array)
        preds = rf_core.predict_tree(X_binned, tree.tree_structure)
        tree_preds.append(preds)

    tree_preds = np.array(tree_preds)

    # Calculate variance across trees for each sample
    prediction_variance = np.var(tree_preds, axis=0)

    # Assert that there is non-zero variance (trees are diverse)
    assert (
        np.mean(prediction_variance) > 1e-4
    ), "Bootstrap failed to induce diversity; trees are identical."


def test_rf_max_features_subsampling(multivariate_data):
    """
    Verifies the feature subsampling mechanism (`max_features`).
    """
    X, y = multivariate_data

    # Force extreme subsampling: only 1 feature considered per split
    model = HistogramRandomForestRegressor(
        n_estimators=10, max_features=1, max_depth=1, random_state=42
    )
    model.fit(X, y)

    root_features = set()
    for tree in model.trees_:
        feature_idx = int(tree.tree_structure[0, 0])
        if feature_idx != -1:
            root_features.add(feature_idx)

    # With 10 trees and random selection, we expect diversity in root features
    assert (
        len(root_features) > 1
    ), "Feature subsampling failed; all trees selected the same root feature."


def test_rf_parallel_execution_consistency(multivariate_data):
    """
    Smoke test for parallel execution via joblib.
    """
    X, y = multivariate_data
    n_estimators = 10

    model = HistogramRandomForestRegressor(
        n_estimators=n_estimators, n_jobs=-1, random_state=42  # Use all available cores
    )
    model.fit(X, y)

    assert (
        len(model.trees_) == n_estimators
    ), f"Parallel fitting produced {len(model.trees_)} trees, expected {n_estimators}."
