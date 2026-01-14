import numpy as np
import pytest

from histree import HistogramGradientBoostingRegressor


@pytest.fixture
def nonlinear_data():
    """Generates a non-linear sine wave dataset."""
    X = np.linspace(0, 10, 200).reshape(-1, 1).astype(np.float32)
    y = np.sin(X).ravel().astype(np.float32)
    return X, y


def test_gb_sequential_residual_reduction(nonlinear_data):
    """
    Verifies that sequential addition of trees reduces the training error.
    """
    X, y = nonlinear_data
    model = HistogramGradientBoostingRegressor(
        n_estimators=5, learning_rate=0.5, random_state=42
    )
    model.fit(X, y)

    current_pred = np.full_like(y, model.initial_prediction_)
    X_binned = model.bin_mapper_.transform(X)

    mse_history = []
    initial_mse = np.mean((y - current_pred) ** 2)
    mse_history.append(initial_mse)

    for tree in model.trees_:
        # W GB drzewa mają metodę predict, bo jest używana w fit
        update = tree.predict(X_binned)
        current_pred += model.learning_rate * update
        mse = np.mean((y - current_pred) ** 2)
        mse_history.append(mse)

    for i in range(1, len(mse_history)):
        assert (
            mse_history[i] < mse_history[i - 1]
        ), f"Training MSE increased at iteration {i}."


def test_gb_early_stopping_logic(nonlinear_data):
    """
    Validates that training halts early when validation score plateaus.
    """
    X, y = nonlinear_data
    X_train, X_val = X[:150], X[150:]
    y_train, y_val = y[:150], y[150:]

    model = HistogramGradientBoostingRegressor(n_estimators=1000, random_state=42)
    # FIX: early_stopping_rounds przekazane do FIT, a nie init
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=5)

    assert len(model.trees_) < 1000, "Early stopping failed to trigger."
    assert hasattr(
        model, "best_iteration_"
    ), "Model missing 'best_iteration_' attribute."
    assert len(model.trees_) == model.best_iteration_ + 1


def test_gb_goss_gradient_retention():
    """
    Unit test for GOSS. Ensures high gradients are retained.
    """
    model = HistogramGradientBoostingRegressor()
    N = 100
    X = np.zeros((N, 1), dtype=np.float32)

    g = np.array([100.0] * 20 + [0.001] * 80, dtype=np.float32)
    h = np.ones(N, dtype=np.float32)

    X_goss, g_goss, h_goss = model._goss(X, g, h, seed=42)

    assert len(g_goss) == 30, f"GOSS returned {len(g_goss)} samples, expected 30."

    high_grad_count = np.sum(g_goss == 100.0)
    assert (
        high_grad_count == 20
    ), f"GOSS discarded high-gradient samples. Found {high_grad_count}, expected 20."
