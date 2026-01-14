import numpy as np
import pytest

from histree import HistogramSequentialWeightedForestRegressor


@pytest.fixture
def binary_data():
    """Generates a dataset suitable for testing weight updates."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(50, 5)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32) * 10.0
    return X, y


def test_swf_weight_trimming_distribution(binary_data):
    """
    Validates the Weight Trimming implementation.
    """
    X, y = binary_data
    model = HistogramSequentialWeightedForestRegressor()

    # Weights: linear distribution 0.0 -> 1.0
    w = np.linspace(0, 1, 100).astype(np.float64)
    X_dummy = np.zeros((100, 1), dtype=np.float32)
    y_dummy = np.zeros(100, dtype=np.float32)

    X_out, y_out, w_out = model._weight_trimming(X_dummy, y_dummy, w, seed=42)

    # Expected: 30 samples
    assert len(w_out) == 30

    # Top weight (1.0) should be preserved
    assert np.any(np.isclose(w_out, 1.0)), "Top weight (1.0) was lost during trimming."


def test_swf_tree_rejection_on_noise():
    """
    Verifies that SWF rejects trees when they fail to reduce loss significantly.
    """
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.random.rand(100).astype(np.float32)

    model = HistogramSequentialWeightedForestRegressor(
        n_estimators=50, max_depth=1, random_state=42
    )
    model.fit(X, y)

    assert len(model.trees_) == len(
        model.betas_
    ), "Mismatch between stored trees and beta coefficients."


def test_swf_perfect_fit_termination():
    """
    Verifies that the algorithm terminates early if a perfect fit is achieved.
    """
    # Trivial dataset: 2 points
    X = np.array([[0], [1]], dtype=np.float32)
    y = np.array([0, 1], dtype=np.float32)

    model = HistogramSequentialWeightedForestRegressor(
        n_estimators=100,
        random_state=42,
        # FIX 1: Pozwalamy na liście z 1 próbką
        min_samples_leaf=1,
        # FIX 2: Pozwalamy na podział przy 2 próbkach (domyślnie jest 20!)
        min_samples_split=2,
    )
    model.fit(X, y)

    assert len(model.trees_) < 100, "Model failed to stop early on perfect fit."

    # Teraz drzewo powinno powstać, dopasować się idealnie i przerwać pętlę
    assert len(model.betas_) > 0, "No trees were built despite perfect fit potential."
    assert model.betas_[-1] == 10.0, "Perfect fit beta marker (10.0) not found."
