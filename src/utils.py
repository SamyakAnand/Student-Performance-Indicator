import math
import os
import sys

import dill
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as exc:
        raise CustomException(exc, sys) from exc


def select_best_model(model_report):
    """Choose the highest finite CV score without imposing an arbitrary R2 cutoff.

    Negative R2 values are valid model-selection results: they indicate that a
    candidate performs worse than a constant-mean baseline, not that training
    itself is invalid. Callers can report the score and decide separately
    whether the model is suitable for a downstream use case.
    """
    finite_scores = {}
    for model_name, score in model_report.items():
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_score):
            finite_scores[model_name] = numeric_score

    if not finite_scores:
        raise ValueError("No finite cross-validation scores were produced.")

    best_model_name = max(finite_scores, key=finite_scores.get)
    return best_model_name, finite_scores[best_model_name]


def evaluate_models(X_train, y_train, models, param, preprocessor):
    """Compare estimators with fold-local preprocessing on training data only.

    Each GridSearchCV candidate is a Pipeline, so the preprocessor is cloned and
    fitted only on that fold's training rows. The returned best estimators are
    refitted on the complete training partition. No holdout data is accepted.
    """
    try:
        report = {}
        best_estimators = {}

        for model_name, model in models.items():
            estimator = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", model),
                ]
            )
            prefixed_params = {
                f"model__{parameter}": values
                for parameter, values in param[model_name].items()
            }
            search = GridSearchCV(
                estimator=estimator,
                param_grid=prefixed_params,
                cv=3,
                scoring="r2",
                refit=True,
            )
            search.fit(X_train, y_train)

            report[model_name] = float(search.best_score_)
            best_estimators[model_name] = search.best_estimator_

        models.update(best_estimators)
        return report

    except Exception as exc:
        raise CustomException(exc, sys) from exc


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as exc:
        raise CustomException(exc, sys) from exc
