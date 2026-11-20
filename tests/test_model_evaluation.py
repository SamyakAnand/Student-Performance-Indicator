import inspect

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.utils import evaluate_models, select_best_model


class FitSizeRecorder(BaseEstimator, TransformerMixin):
    fit_sizes = []

    def fit(self, X, y=None):
        type(self).fit_sizes.append(len(X))
        return self

    def transform(self, X):
        return np.asarray(X)


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
        preprocessor=StandardScaler(),
    )

    assert set(report) == {"linear", "mean_baseline"}
    assert all(np.isfinite(score) for score in report.values())
    assert report["linear"] > report["mean_baseline"]
    assert hasattr(models["linear"].named_steps["model"], "coef_")
    assert hasattr(models["mean_baseline"].named_steps["model"], "constant_")
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
        preprocessor=StandardScaler(),
    )

    best_pipeline = models["ridge"]
    assert np.isfinite(report["ridge"])
    assert best_pipeline.named_steps["model"].alpha in parameters["ridge"]["alpha"]
    assert best_pipeline.named_steps["model"].coef_.shape == (features.shape[1],)
    assert "preprocessor" in best_pipeline.named_steps


def test_preprocessor_is_fitted_inside_each_cross_validation_fold():
    features, target = make_regression(
        n_samples=90,
        n_features=5,
        noise=10.0,
        random_state=11,
    )
    FitSizeRecorder.fit_sizes.clear()
    models = {"ridge": Ridge()}
    parameters = {"ridge": {"alpha": [0.1, 1.0]}}

    evaluate_models(
        X_train=features,
        y_train=target,
        models=models,
        param=parameters,
        preprocessor=FitSizeRecorder(),
    )

    # Three-fold CV fits on 60 rows per fold, then refits the winner on all 90.
    assert FitSizeRecorder.fit_sizes.count(60) >= 3
    assert 90 in FitSizeRecorder.fit_sizes


def test_best_model_selection_accepts_scores_below_arbitrary_threshold():
    name, score = select_best_model(
        {"weak_but_finite": 0.42, "lower_score": 0.31}
    )

    assert name == "weak_but_finite"
    assert score == pytest.approx(0.42)


def test_best_model_selection_ignores_non_finite_scores():
    name, score = select_best_model(
        {"nan_model": np.nan, "infinite_model": np.inf, "finite_model": -0.2}
    )

    assert name == "finite_model"
    assert score == pytest.approx(-0.2)


def test_best_model_selection_requires_at_least_one_finite_score():
    with pytest.raises(ValueError, match="No finite cross-validation scores"):
        select_best_model({"nan_model": np.nan, "infinite_model": np.inf})
