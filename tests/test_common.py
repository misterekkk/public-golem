import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from histree import (
    HistogramGradientBoostingRegressor,
    HistogramRandomForestRegressor,
    HistogramSequentialWeightedForestRegressor,
)

# --- CONFIGURATIONS ---

# Configuration for Convergence Test (Sanity Check)
# We need models to be "unrestrained" to perfectly memorize data.
# RF needs bootstrap=False to behave deterministically on training data.
CONVERGENCE_CONFIGS = [
    (
        HistogramGradientBoostingRegressor,
        {
            "reg_lambda": 0.0,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_depth": 10,
            "n_bins": 64,
        },
    ),
    (
        HistogramRandomForestRegressor,
        {
            "bootstrap": False,
            "max_features": 1.0,  # Disable randomness for perfect fit
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_depth": 10,
            "n_bins": 64,
        },
    ),
    (
        HistogramSequentialWeightedForestRegressor,
        {
            "reg_lambda": 0.0,
            "subsample": 1.0,  # Disable subsampling
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_depth": 10,
            "n_bins": 64,
        },
    ),
]

# Configuration for Stochasticity Test (Random State Impact)
# We need to explicitly ENABLE randomness (bootstrap, subsample, colsample)
# so that changing the seed actually changes the model.
STOCHASTIC_CONFIGS = [
    (HistogramGradientBoostingRegressor, {"subsample": 0.5}),
    (HistogramRandomForestRegressor, {"bootstrap": True, "max_features": "sqrt"}),
    (HistogramSequentialWeightedForestRegressor, {"subsample": 0.5}),
]

# Basic list for API checks where params don't matter much
ALL_MODELS = [
    HistogramGradientBoostingRegressor,
    HistogramRandomForestRegressor,
    HistogramSequentialWeightedForestRegressor,
]

# --- FIXTURES ---


@pytest.fixture
def linear_toy_dataset():
    """Provides a trivial linear dataset (y = 2x + 1)."""
    X = np.array([[1], [2], [3], [4], [5], [10]], dtype=np.float32)
    y = np.array([3, 5, 7, 9, 11, 21], dtype=np.float32)
    return X, y


@pytest.fixture
def synthetic_random_dataset():
    """Provides a synthetic dataset with uniform distribution."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 50, 5
    X = rng.rand(n_samples, n_features).astype(np.float32)
    y = rng.rand(n_samples).astype(np.float32)
    return X, y


# --- TEST SUITE ---


@pytest.mark.parametrize("ModelClass, params", CONVERGENCE_CONFIGS)
def test_estimator_convergence_on_trivial_data(ModelClass, params, linear_toy_dataset):
    """
    Sanity Check: Verifies that the estimator can overfit a trivial dataset
    GIVEN appropriate hyperparameters (e.g. no regularization, no bootstrap).
    """
    X, y = linear_toy_dataset

    # Initialize model with specific params for overfitting
    model = ModelClass(random_state=42, **params)
    model.fit(X, y)

    predictions = model.predict(X)

    np.testing.assert_allclose(
        predictions,
        y,
        rtol=1e-3,
        atol=1e-3,
        err_msg=f"{ModelClass.__name__} failed to converge with params: {params}",
    )


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_prediction_determinism(ModelClass, synthetic_random_dataset):
    """
    Reproducibility Check: Ensures that estimators initialized with a fixed
    random state yield bitwise-identical predictions.
    """
    X, y = synthetic_random_dataset
    seed = 123

    # Run 1
    model_1 = ModelClass(random_state=seed, n_bins=64)
    model_1.fit(X, y)
    preds_1 = model_1.predict(X)

    # Run 2
    model_2 = ModelClass(random_state=seed, n_bins=64)
    model_2.fit(X, y)
    preds_2 = model_2.predict(X)

    np.testing.assert_array_equal(
        preds_1,
        preds_2,
        err_msg=f"{ModelClass.__name__} violates determinism contract.",
    )


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
def test_api_contract_output_shapes(ModelClass, synthetic_random_dataset):
    """
    API Compliance: Validates inference output types and shapes.
    """
    X, y = synthetic_random_dataset
    model = ModelClass(random_state=42)
    model.fit(X, y)

    predictions = model.predict(X)

    assert isinstance(
        predictions, np.ndarray
    ), f"Expected numpy.ndarray, got {type(predictions)}"
    assert (
        predictions.shape == y.shape
    ), f"Shape mismatch: {predictions.shape} vs {y.shape}"
    assert np.isfinite(predictions).all(), "Inference produced non-finite values."


@pytest.mark.parametrize("ModelClass, params", STOCHASTIC_CONFIGS)
def test_random_state_impact(ModelClass, params, synthetic_random_dataset):
    """
    Stochasticity Verification: Ensures that varying the random seed produces
    distinct models when stochastic features (bootstrap/subsample) are enabled.
    """
    X, y = synthetic_random_dataset

    # Initialize models with distinct seeds AND stochastic params enabled
    model_1 = ModelClass(random_state=1, **params)
    model_2 = ModelClass(random_state=2, **params)

    model_1.fit(X, y)
    model_2.fit(X, y)

    preds_1 = model_1.predict(X)
    preds_2 = model_2.predict(X)

    # Assert divergence
    assert not np.allclose(
        preds_1, preds_2
    ), f"{ModelClass.__name__} appears insensitive to random_state even with {params}."


@pytest.mark.parametrize("ModelClass", ALL_MODELS)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_sklearn_ecosystem_compatibility(ModelClass, synthetic_random_dataset):
    """
    Integration Test: Verifies compatibility with Scikit-Learn ecosystem tools
    like Pipeline and GridSearchCV.
    """
    X, y = synthetic_random_dataset

    # 1. Define a Pipeline
    # Standard scaling -> Your Regressor
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", ModelClass(random_state=42, n_bins=32)),
        ]
    )

    # 2. Define Hyperparameter Grid
    # We test changing 'n_bins' via the pipeline parameter naming convention
    param_grid = {"regressor__n_bins": [16, 64], "regressor__min_samples_leaf": [5]}

    # 3. Run GridSearchCV
    # This checks: cloning, set_params, get_params, fit, predict, and scoring
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=2,
        scoring="neg_mean_squared_error",
        n_jobs=1,  # Keep it simple for unit tests
    )

    search.fit(X, y)

    # 4. Verification
    assert search.best_params_ is not None
    assert search.best_estimator_ is not None

    # Ensure prediction works on the fitted search object
    preds = search.predict(X)
    assert preds.shape == y.shape
