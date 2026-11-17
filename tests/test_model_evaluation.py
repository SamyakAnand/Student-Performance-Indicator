import inspect

import numpy as np
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge

from src.utils import evaluate_models


def test_model_selection_uses_training_cross_validation_only():
    features, target = make_regression(
        n_samples=90,
        n_features=5,
        noise=8.0,
        random_state=42,
    )
    models = {
        "linear": LinearRegression(),
        "mean_baseline": DummyRegressor(strategy="mean"),
    }
    parameters = {
        "linear": {},
        "mean_baseline": {},
    }

    report = evaluate_models(
        X_train=features,
        y_train=target,
        models=models,
        param=parameters,
    )

    assert set(report) == {"linear", "mean_baseline"}
    assert all(np.isfinite(score) for score in report.values())
    assert report["linear"] > report["mean_baseline"]
    assert hasattr(models["linear"], "coef_")
    assert hasattr(models["mean_baseline"], "constant_")
    assert "X_test" not in inspect.signature(evaluate_models).parameters
    assert "y_test" not in inspect.signature(evaluate_models).parameters


def test_grid_search_best_estimator_is_refitted_on_training_partition():
    features, target = make_regression(
        n_samples=75,
        n_features=4,
        noise=15.0,
        random_state=7,
    )
    models = {"ridge": Ridge()}
    parameters = {"ridge": {"alpha": [0.01, 0.1, 1.0, 10.0]}}

    report = evaluate_models(
        X_train=features,
        y_train=target,
        models=models,
        param=parameters,
    )

    assert np.isfinite(report["ridge"])
    assert models["ridge"].alpha in parameters["ridge"]["alpha"]
    assert models["ridge"].coef_.shape == (features.shape[1],)
